// Build and serialize metadata v7 without materializing Python descriptors.
// Layout and ordering reference: dcp_mega_metadata.py.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

namespace py = pybind11;
namespace {
using Ints = std::vector<int32_t>;
using Row = std::array<int32_t, 8>;
using Rows = std::vector<Row>;
int32_t ceil_div(int32_t a, int32_t b) { return (a + b - 1) / b; }

Ints build_packed_queues(const Ints& cu, int32_t heads, int32_t world,
                        int32_t sms, int32_t comm, int32_t block_n,
                        const Ints& chunk_splits, const Ints& history_splits,
                        int32_t copy_vectors, const std::optional<Ints>& history_blocks) {
    const int32_t total_q = cu.back(), token_blocks = ceil_div(total_q, 16);
    const int32_t ctas = sms - comm;
    const int32_t final_tokens = token_blocks < ctas ? 4 : token_blocks < 2 * ctas ? 8 : 16;
    Rows attention, publish, combine, final;
    Ints q_tasks, q_deps, publish_deps, final_deps;
    std::array<Ints, 2> bases{Ints(cu.size() - 1), Ints(cu.size() - 1)};
    for (int32_t kind = 0; kind < 2; ++kind) {
        const auto& splits = kind ? history_splits : chunk_splits;
        const int32_t domain_heads = kind ? world * heads : heads;
        for (size_t b = 0; b + 1 < cu.size(); ++b) {
            bases[kind][b] = attention.size();
            int32_t length = cu[b + 1] - cu[b];
            for (int32_t m = 0; m < ceil_div(length * domain_heads, 128); ++m) {
                int32_t dep_begin = q_deps.size();
                if (kind) {
                    int32_t first = (cu[b] + m * 128 / domain_heads) / 16;
                    int32_t last = (cu[b] + (std::min((m + 1) * 128, length * domain_heads) - 1) / domain_heads) / 16;
                    for (int32_t d = first; d <= last; ++d) q_deps.push_back(d);
                }
                int32_t dep_count = q_deps.size() - dep_begin;
                for (int32_t s = 0; s < splits[b]; ++s)
                    attention.push_back({kind, static_cast<int32_t>(b), m, 0, s,
                                         dep_begin, dep_count, static_cast<int32_t>(attention.size())});
            }
        }
    }
    const int32_t chunk_count = bases[1][0];
    Ints order(token_blocks);
    std::iota(order.begin(), order.end(), 0);
    if (history_blocks) {
        auto work = [&](const Row& row) {
            int32_t n = (*history_blocks)[row[1]], splits = history_splits[row[1]];
            int32_t width = ceil_div(n, splits);
            return std::max(0, std::min(width, n - row[4] * width)) + 4;
        };
        std::vector<double> unlock(token_blocks, 0.0);
        for (const auto& row : attention) if (row[0]) {
            double share = static_cast<double>(work(row)) / row[6];
            for (int32_t d = row[5]; d < row[5] + row[6]; ++d) unlock[q_deps[d]] += share;
        }
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return unlock[a] != unlock[b] ? unlock[a] > unlock[b] : a < b;
        });
        Ints position(token_blocks);
        for (int32_t i = 0; i < token_blocks; ++i) position[order[i]] = i;
        auto key = [&](const Row& row) {
            int32_t release = 0;
            for (int32_t d = row[5]; d < row[5] + row[6]; ++d)
                release = std::max(release, position[q_deps[d]]);
            return std::make_tuple(release / std::max(1, comm / world), -work(row),
                                   row[1], row[2], row[4], row[7]);
        };
        std::sort(attention.begin() + chunk_count, attention.end(),
                  [&](const Row& a, const Row& b) { return key(a) < key(b); });
    }
    for (int32_t tile : order) for (int32_t rank = 0; rank < world; ++rank) {
        q_tasks.insert(q_tasks.end(), {rank, 0, tile * 16, std::min(16, total_q - tile * 16)});
    }

    // A physical vector's completion IDs are a contiguous split range at its
    // packed M tile. Derive that range instead of constructing a vector->IDs map.
    auto dependencies = [&](int32_t kind, int32_t begin, int32_t end, int32_t rank) {
        Ints deps;
        int32_t domain_heads = kind ? world * heads : heads;
        const auto& splits = kind ? history_splits : chunk_splits;
        while (begin < end) {
            int32_t token = begin / heads, head = begin % heads;
            size_t b = std::upper_bound(cu.begin(), cu.end(), token) - cu.begin() - 1;
            int32_t count = std::min(end - begin, kind ? heads - head : cu[b + 1] * heads - begin);
            int32_t packed = (token - cu[b]) * domain_heads + (kind ? rank * heads : 0) + head;
            int32_t first = bases[kind][b] + packed / 128 * splits[b];
            int32_t last = bases[kind][b] + ((packed + count - 1) / 128 + 1) * splits[b];
            for (int32_t id = first; id < last; ++id) deps.push_back(id);
            begin += count;
        }
        std::sort(deps.begin(), deps.end());
        deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        return deps;
    };

    std::vector<Rows> combine_by_publish;
    for (int32_t rank = 0; rank < world; ++rank) {
        for (int32_t tile = 0; tile < token_blocks; ++tile) {
            int32_t begin = tile * 16 * heads, end = std::min((tile + 1) * 16, total_q) * heads;
            int32_t pub_id = publish.size(), dep_begin = publish_deps.size();
            Rows tasks;
            int32_t region = begin;
            while (region < end) {
                size_t b = std::upper_bound(cu.begin(), cu.end(), region / heads) - cu.begin() - 1;
                int32_t region_end = std::min(end, cu[b + 1] * heads);
                int32_t step = history_splits[b] > 1 ? 1 : copy_vectors;
                for (int32_t vector = region; vector < region_end; vector += step) {
                    int32_t valid = std::min(step, region_end - vector);
                    Ints deps = dependencies(1, vector, vector + valid, rank);
                    tasks.push_back({pub_id, vector, valid, static_cast<int32_t>(publish_deps.size()),
                                     static_cast<int32_t>(deps.size()), static_cast<int32_t>(b), history_splits[b], 0});
                    publish_deps.insert(publish_deps.end(), deps.begin(), deps.end());
                }
                region = region_end;
            }
            publish.push_back({rank, begin, end - begin, dep_begin,
                               static_cast<int32_t>(publish_deps.size()) - dep_begin, 1,
                               static_cast<int32_t>(tasks.size()), 0});
            combine_by_publish.push_back(std::move(tasks));
        }
    }
    for (int32_t tile : order) for (int32_t rank = 0; rank < world; ++rank) {
        const auto& tasks = combine_by_publish[rank * token_blocks + tile];
        combine.insert(combine.end(), tasks.begin(), tasks.end());
    }
    for (int32_t tile : order) {
        int32_t valid_tokens = std::min(16, total_q - tile * 16);
        for (int32_t offset = 0; offset < valid_tokens; offset += final_tokens) {
            int32_t begin = (tile * 16 + offset) * heads;
            int32_t valid = std::min(final_tokens, valid_tokens - offset) * heads;
            Ints deps = dependencies(0, begin, begin + valid, 0);
            final.push_back({begin, valid, static_cast<int32_t>(final_deps.size()),
                             static_cast<int32_t>(deps.size()), tile, 0, 0, 0});
            final_deps.insert(final_deps.end(), deps.begin(), deps.end());
        }
    }

    Ints payload(40, 0);
    auto append_rows = [&](const Rows& rows) {
        int32_t offset = payload.size();
        for (const auto& row : rows) payload.insert(payload.end(), row.begin(), row.end());
        return offset;
    };
    auto append_ints = [&](const Ints& ints) {
        int32_t offset = payload.size();
        payload.insert(payload.end(), ints.begin(), ints.end());
        return offset;
    };
    int32_t attention_offset = append_rows(attention);
    int32_t q_offset = append_ints(q_tasks), q_deps_offset = append_ints(q_deps);
    int32_t publish_offset = append_rows(publish), combine_offset = append_rows(combine);
    int32_t publish_deps_offset = append_ints(publish_deps), final_offset = append_rows(final);
    int32_t final_deps_offset = append_ints(final_deps);
    int32_t chunk_splits_offset = append_ints(chunk_splits), history_splits_offset = append_ints(history_splits);
    int32_t max_chunk = *std::max_element(chunk_splits.begin(), chunk_splits.end());
    int32_t max_history = *std::max_element(history_splits.begin(), history_splits.end());
    int32_t max_splits = std::max(max_chunk, max_history);
    Ints header{
        7, static_cast<int32_t>(attention.size()), static_cast<int32_t>(q_tasks.size() / 4),
        static_cast<int32_t>(q_deps.size()), static_cast<int32_t>(publish.size()),
        static_cast<int32_t>(publish_deps.size()), static_cast<int32_t>(final.size()),
        static_cast<int32_t>(final_deps.size()), chunk_count, static_cast<int32_t>(attention.size()) - chunk_count,
        total_q, total_q * heads, max_splits, max_chunk, max_history, 1, max_splits > 1, block_n,
        static_cast<int32_t>(chunk_splits.size()), 1, 2, attention_offset, q_offset, q_deps_offset,
        publish_offset, publish_deps_offset, final_offset, final_deps_offset,
        chunk_splits_offset, history_splits_offset, static_cast<int32_t>(payload.size()), 0, 1,
        token_blocks, token_blocks, token_blocks * (world - 1), token_blocks * (world - 1), world,
        static_cast<int32_t>(combine.size()), combine_offset};
    std::copy(header.begin(), header.end(), payload.begin());
    return payload;
}
}  // namespace

void bind_dcp_mega_metadata(py::module_& m) {
    m.def("build_packed_queues", [](const Ints& cu, int32_t heads, int32_t world,
            int32_t sms, int32_t comm, int32_t block_n, const Ints& chunk_splits,
            const Ints& history_splits, int32_t copy_vectors,
            const std::optional<Ints>& history_blocks) {
        Ints payload;
        {
            py::gil_scoped_release release;
            payload = build_packed_queues(cu, heads, world, sms, comm, block_n,
                                          chunk_splits, history_splits, copy_vectors, history_blocks);
        }
        return py::bytes(reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(int32_t));
    });
}

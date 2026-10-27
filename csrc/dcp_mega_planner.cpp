// CPU implementation of dcp_mega_metadata's critical-wave search and cost model.
// No CUDA calls: the caller can prepare the next batch while a graph is running.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <vector>

namespace py = pybind11;
void bind_dcp_mega_metadata(py::module_& m);
namespace {
using I = int64_t;
using Vec = std::vector<I>;
// Cost model constants mirror dcp_mega_metadata.py (HEURISTIC_*).
constexpr I task_overhead = 32;
constexpr double bandwidth_floor = 0.43;
constexpr double combine_overhead = 2.5;
constexpr double partial_vector_cost = 0.47;
constexpr I compute_warps = 12;
constexpr double no_cutoff = std::numeric_limits<double>::infinity();
constexpr double min_gain = 0.10;
I ceil_div(I a, I b) { return (a + b - 1) / b; }
I split_work(I n, I s, I i) {
    I width = ceil_div(n, s);
    return std::max<I>(0, std::min(width, n - i * width)) + task_overhead;
}
template <typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

struct Plan {
    Vec splits;
    std::string source;
    double attention_span = 0;
    I attention_tasks = 0;
    I combine_tasks = 0, partial_vectors = 0;
    double combine_work = 0, penalty = 0;
    Vec critical;
    double score = 0;
    auto key() const {
        return std::make_tuple(score, attention_span, attention_tasks,
                              std::accumulate(splits.begin(), splits.end(), I{0}), splits);
    }
    py::tuple as_tuple() const {
        return py::make_tuple(splits, source, attention_span, attention_tasks,
                              combine_tasks, partial_vectors, combine_work,
                              penalty, critical, score);
    }
};

struct Task {
    I batch, m, split, id, work, dep_begin, dep_end;
    bool history;
};

struct CombineProfile { I tasks = 0, vectors = 0; double work = 0; };
struct Stats {
    I candidates = 0, bound_pruned = 0, attention_pruned = 0;
    I combine_pruned = 0, scored = 0;
};

struct Planner {
    Vec q, history, chunk, legacy, native_chunk, cu;
    I heads, world, block_n, ctas, comm, max_splits;
    bool legacy_native, reorder_no_split, reorder_split, prune;
    Stats stats;

    CombineProfile combine_profile(const Vec& splits, I copy) const {
        CombineProfile p;
        for (size_t b = 0; b < q.size(); ++b) {
            for (I tile = cu[b] / 16 * 16; tile < cu[b + 1]; tile += 16) {
                I vectors = (std::min(tile + 16, cu[b + 1]) - std::max(tile, cu[b])) * heads;
                p.tasks += splits[b] > 1 ? vectors : ceil_div(vectors, copy);
                p.vectors += vectors * splits[b];
            }
        }
        p.tasks *= world;
        p.vectors *= world;
        p.work = combine_overhead * p.tasks + partial_vector_cost * p.vectors;
        return p;
    }

    std::optional<Plan> profile(const Vec& splits, const std::string& source,
                                double cutoff = no_cutoff) {
        ++stats.candidates;
        const Vec& chunks = legacy_native && source == "legacy_dynamic" && !native_chunk.empty()
            ? native_chunk : chunk;
        bool reorder = legacy_native && source == "legacy_dynamic" ? false
            : (*std::max_element(splits.begin(), splits.end()) > 1 ? reorder_split : reorder_no_split);
        I copy = 1;
        std::set<I> copies{1, heads, 2 * heads, 4 * heads, std::min<I>(8 * heads, 32)};
        CombineProfile combine;
        for (auto it = copies.rbegin(); it != copies.rend(); ++it) {
            combine = combine_profile(splits, *it);
            if (combine.tasks >= static_cast<I>(std::ceil(0.8 * ctas * compute_warps)) || *it == 1) {
                copy = *it;
                break;
            }
        }
        // Lower bounds on the final makespan: attention work cannot finish
        // faster than full bandwidth, no task runs faster than the floor, and
        // a CTA's attention (at least floor time per unit) occupies all of its
        // combine warps. Dependencies can only increase these bounds. Pruning at
        // cutoff preserves strict improvement; the slack absorbs rounding.
        if (prune && cutoff != no_cutoff) {
            I work = 0, longest = 0;
            for (size_t b = 0; b < q.size(); ++b) {
                for (I m = 0; m < ceil_div(q[b] * heads, 128); ++m) {
                    I n = ceil_div(std::min(q[b], ceil_div((m + 1) * 128, heads)), block_n);
                    work += n + chunks[b] * task_overhead;
                    longest = std::max(longest, split_work(n, chunks[b], 0));
                }
                work += ceil_div(q[b] * heads * world, 128) * (history[b] + splits[b] * task_overhead);
                longest = std::max(longest, split_work(history[b], splits[b], 0));
            }
            double lower_bound = std::max({
                bandwidth_floor * longest, static_cast<double>(work) / ctas,
                (bandwidth_floor * work * compute_warps + combine.work) / (ctas * compute_warps)});
            if (lower_bound * (1 - 1e-9) >= cutoff) { ++stats.bound_pruned; return std::nullopt; }
        }

        std::vector<Task> tasks;
        Vec bases(q.size());
        for (bool history_domain : {false, true}) {
            I domain_heads = heads * (history_domain ? world : 1);
            const Vec& domain_splits = history_domain ? splits : chunks;
            for (size_t b = 0; b < q.size(); ++b) {
                if (history_domain) bases[b] = tasks.size();
                for (I m = 0; m < ceil_div(q[b] * domain_heads, 128); ++m) {
                    I n = history_domain ? history[b]
                        : ceil_div(std::min(q[b], ceil_div((m + 1) * 128, heads)), block_n);
                    I begin = (cu[b] + m * 128 / domain_heads) / 16;
                    I end = (cu[b] + (std::min((m + 1) * 128, q[b] * domain_heads) - 1) / domain_heads) / 16 + 1;
                    for (I s = 0; s < domain_splits[b]; ++s) {
                        tasks.push_back({static_cast<I>(b), m, s, static_cast<I>(tasks.size()),
                                         split_work(n, domain_splits[b], s), begin, end, history_domain});
                    }
                }
            }
        }
        if (reorder) {
            std::vector<double> unlock(ceil_div(cu.back(), 16), 0.0);
            for (const auto& t : tasks) if (t.history) {
                for (I d = t.dep_begin; d < t.dep_end; ++d)
                    unlock[d] += static_cast<double>(t.work) / (t.dep_end - t.dep_begin);
            }
            Vec order(unlock.size()), position(unlock.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](I a, I b) {
                return unlock[a] != unlock[b] ? unlock[a] > unlock[b] : a < b;
            });
            for (size_t i = 0; i < order.size(); ++i) position[order[i]] = i;
            auto key = [&](const Task& t) {
                I release = 0;
                for (I d = t.dep_begin; d < t.dep_end; ++d) release = std::max(release, position[d]);
                return std::make_tuple(release / std::max<I>(1, comm / world), -t.work,
                                       t.batch, t.m, t.split, t.id);
            };
            auto first = std::find_if(tasks.begin(), tasks.end(), [](const Task& t) { return t.history; });
            std::sort(first, tasks.end(), [&](const Task& a, const Task& b) { return key(a) < key(b); });
        }

        // The first CTA to become free takes the next task. Running CTAs advance
        // together: track progress in work units and convert it to time between
        // completions with the current number of running CTAs.
        MinHeap<std::tuple<I, I, I>> running;
        Vec last_sequence(ctas, -1);
        std::vector<double> finish_times(ctas, 0.0), completions(tasks.size(), 0.0);
        size_t next_task = 0;
        I progress = 0;
        double now = 0;
        auto start = [&](I worker) {
            if (next_task == tasks.size()) return;
            const Task& task = tasks[next_task++];
            last_sequence[worker] = task.history ? task.batch : -1;
            running.emplace(progress + task.work, worker, task.id);
        };
        for (I i = 0; i < ctas; ++i) start(i);
        while (!running.empty()) {
            I finish = std::get<0>(running.top());
            now += static_cast<double>(finish - progress) *
                std::max(bandwidth_floor, static_cast<double>(running.size()) / ctas);
            progress = finish;
            Vec finished;
            while (!running.empty() && std::get<0>(running.top()) == finish) {
                auto [_, worker, id] = running.top();
                running.pop();
                completions[id] = now;
                finish_times[worker] = now;
                finished.push_back(worker);
            }
            if (prune && now >= cutoff) { ++stats.attention_pruned; return std::nullopt; }
            for (I worker : finished) start(worker);
        }
        double attention_span = now;
        std::set<I> critical;
        for (I i = 0; i < ctas; ++i)
            if (finish_times[i] == attention_span && last_sequence[i] >= 0) critical.insert(last_sequence[i]);

        // Worker identities do not affect combine costs. Preserve FIFO repeats,
        // grouping equal availability just as the Python reference does.
        MinHeap<std::pair<double, I>> workers;
        std::vector<double> sorted_finish = finish_times;
        std::sort(sorted_finish.begin(), sorted_finish.end());
        for (size_t i = 0; i < sorted_finish.size();) {
            size_t j = i + 1;
            while (j < sorted_finish.size() && sorted_finish[j] == sorted_finish[i]) ++j;
            workers.emplace(sorted_finish[i], (j - i) * compute_warps);
            i = j;
        }
        double score = attention_span;
        size_t batch = 0;
        for (I tile = 0; tile < cu.back(); tile += 16) {
            I tile_end = std::min(tile + 16, cu.back());
            while (cu[batch + 1] <= tile) ++batch;
            for (I rank = 0; rank < world; ++rank) {
                size_t b = batch;
                I vector = tile * heads;
                while (vector < tile_end * heads) {
                    I region_end = std::min(tile_end, cu[b + 1]) * heads;
                    I step = splits[b] > 1 ? 1 : copy;
                    while (vector < region_end) {
                        I valid = std::min(step, region_end - vector);
                        I token = vector / heads, head = vector % heads;
                        I packed = (token - cu[b]) * heads * world + rank * heads + head;
                        I repeats = 1;
                        double ready = 0;
                        if (step == 1) {
                            repeats = std::min({region_end - vector, heads - head, 128 - packed % 128});
                            I first = bases[b] + packed / 128 * splits[b];
                            for (I s = 0; s < splits[b]; ++s) ready = std::max(ready, completions[first + s]);
                        } else {
                            for (I p = vector; p < vector + valid;) {
                                I count = std::min(vector + valid - p, heads - p % heads);
                                I offset = (p / heads - cu[b]) * heads * world + rank * heads + p % heads;
                                for (I d = offset / 128; d <= (offset + count - 1) / 128; ++d)
                                    ready = std::max(ready, completions[bases[b] + d]);
                                p += count;
                            }
                        }
                        I remaining = repeats;
                        while (remaining) {
                            auto [available, count] = workers.top(); workers.pop();
                            I used = std::min(remaining, count);
                            double finish = std::max(available, ready) +
                                (combine_overhead + partial_vector_cost * valid * splits[b]);
                            if (used < count) workers.emplace(available, count - used);
                            workers.emplace(finish, used);
                            score = std::max(score, finish);
                            if (prune && score >= cutoff) { ++stats.combine_pruned; return std::nullopt; }
                            remaining -= used;
                        }
                        vector += valid * repeats;
                    }
                    ++b;
                }
            }
        }
        ++stats.scored;
        return Plan{splits, source, attention_span, static_cast<I>(tasks.size()),
                    combine.tasks, combine.vectors, combine.work, score - attention_span,
                    Vec(critical.begin(), critical.end()), score};
    }

    std::pair<Plan, Plan> run() {
        cu = {0};
        for (I length : q) cu.push_back(cu.back() + length);
        Plan baseline = *profile(Vec(q.size(), 1), "nosplit");
        Vec caps = legacy;
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i] <= 16 && history[i] >= ctas) caps[i] = std::min(max_splits, std::max<I>(caps[i], 4));
        Plan iterative = baseline;
        while (true) {
            std::set<Vec> vectors;
            for (size_t i = 0; i < q.size(); ++i) if (iterative.splits[i] < caps[i]) {
                Vec next = iterative.splits; ++next[i]; vectors.insert(next);
            }
            if (iterative.critical.size() > 1 && std::all_of(iterative.critical.begin(), iterative.critical.end(),
                    [&](I i) { return iterative.splits[i] < caps[i]; })) {
                Vec next = iterative.splits;
                for (I i : iterative.critical) ++next[i];
                vectors.insert(next);
            }
            std::optional<Plan> best;
            for (const auto& splits : vectors) {
                // Retain ties with best for the original tie-break.
                double cutoff = best && best->score < iterative.score
                    ? std::nextafter(best->score, no_cutoff) : iterative.score;
                auto candidate = profile(splits, "iterative", cutoff);
                if (candidate && (!best || candidate->key() < best->key())) best = std::move(candidate);
            }
            if (!best || best->score >= iterative.score) break;
            iterative = *best;
        }
        Plan selected = baseline;
        if (iterative.key() < selected.key()) selected = iterative;
        if (legacy != baseline.splits) {
            Plan native = *profile(legacy, "legacy_dynamic");
            if (native.key() < selected.key()) selected = native;
        }
        double gain = (baseline.score - selected.score) / baseline.score;
        if (gain < min_gain || selected.score >= baseline.score) selected = baseline;
        return {baseline, selected};
    }
};
}  // namespace

PYBIND11_MODULE(_dcp_mega_planner, m) {
    bind_dcp_mega_metadata(m);
    m.def("critical_wave_plan", [](Vec q, Vec history, I heads, I world, I block_n,
            I sms, I comm, Vec chunk, Vec legacy, I max_splits,
            std::optional<Vec> native_chunk, bool legacy_native,
            bool reorder_no_split, bool reorder_split, bool prune) {
        Planner planner{std::move(q), std::move(history), std::move(chunk), std::move(legacy),
                        native_chunk.value_or(Vec{}), {}, heads, world, block_n, sms - comm,
                        comm, max_splits, legacy_native, reorder_no_split, reorder_split, prune, {}};
        std::pair<Plan, Plan> plans;
        {
            py::gil_scoped_release release;
            plans = planner.run();
        }
        const auto& s = planner.stats;
        py::dict stats;
        stats["candidates"] = s.candidates;
        stats["bound_pruned"] = s.bound_pruned;
        stats["attention_pruned"] = s.attention_pruned;
        stats["combine_pruned"] = s.combine_pruned;
        stats["scored"] = s.scored;
        return py::make_tuple(plans.first.as_tuple(), plans.second.as_tuple(), stats);
    }, py::arg("q_lengths"), py::arg("history_n_blocks"), py::arg("hq_local"),
       py::arg("dcp_size"), py::arg("block_n"), py::arg("num_sms"), py::arg("num_comm_sm"),
       py::arg("chunk_sequence_splits"), py::arg("legacy_splits"), py::arg("max_num_splits"),
       py::arg("native_chunk_sequence_splits"), py::arg("legacy_is_native"),
       py::arg("reorder_no_split"), py::arg("reorder_split"), py::arg("prune") = true);
}

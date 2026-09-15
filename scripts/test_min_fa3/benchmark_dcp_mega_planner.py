"""Compare cold Python/C++ searches and metadata builds on the same trace batches."""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import dcp_mega_metadata as metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    options = parser.parse_args()
    requests = json.loads(options.workload.read_text())["measured"]
    native = metadata._native_critical_wave_plan
    if native is None:
        raise RuntimeError("Build the C++ CPU planner with make first")
    results = []

    def measure(fn):
        samples = []
        for _ in range(options.repeats):
            start = time.perf_counter()
            value = fn()
            samples.append((time.perf_counter() - start) * 1000)
        return value, dict(median_ms=statistics.median(samples), samples_ms=samples)

    for batch in (4, 16, 64):
        q = tuple(range(batch + 1))
        h = (0, *itertools.accumulate((r["history_tokens"] + 3) // 4 for r in requests[:batch]))
        # FIFO ablation and the actual serving default (automatic history order).
        for reorder in (False, None):
            config = dict(hq_local=8, dcp_size=4, num_sms=78, num_comm_sm=8,
                          max_num_splits=8, block_n_override=128,
                          scheduler_heuristic=None, reorder_history_override=reorder)
            captured = {}

            def record(q_lengths, history_n_blocks, **kwargs):
                captured.update(q_lengths=q_lengths, history_n_blocks=history_n_blocks, **kwargs)
                return native(q_lengths, history_n_blocks, **kwargs)

            metadata._critical_wave_plan_cached.cache_clear()
            metadata._metadata_queue_cache.clear()
            with patch.object(metadata, "_native_critical_wave_plan", record):
                expected_metadata = metadata.build_dcp_mega_metadata(q, h, **config)
            expected_plan = metadata._critical_wave_plan_python(**captured)
            expected_payload = metadata.pack_dcp_mega_metadata(expected_metadata, pre_phase=1, post_phase=2)

            for version in ("python", "cpp", "cpp_pruned"):
                prune = version == "cpp_pruned"

                def cpp(q_lengths, history_n_blocks, **kwargs):
                    return native(q_lengths, history_n_blocks, **kwargs, prune=prune)

                if version == "python":
                    plans, search = measure(lambda: metadata._critical_wave_plan_python(**captured))
                    stats = None
                else:
                    raw, search = measure(lambda: native(**captured, prune=prune))
                    plans = tuple(metadata._CriticalWavePlan(
                        tuple(p[0]), *p[1:8], tuple(p[8]), p[9]
                    ) for p in raw[:2])
                    stats = raw[2]
                assert plans == expected_plan

                def cold_build():
                    metadata._critical_wave_plan_cached.cache_clear()
                    metadata._metadata_queue_cache.clear()
                    return metadata.build_dcp_mega_metadata(q, h, **config)

                with patch.object(metadata, "_native_critical_wave_plan", None if version == "python" else cpp):
                    value, build = measure(cold_build)
                assert value == expected_metadata
                payload, pack = measure(lambda: metadata.pack_dcp_mega_metadata(value, pre_phase=1, post_phase=2))
                assert payload == expected_payload
                row = dict(batch=batch, reorder_history=reorder, version=version,
                           search=search, cold_build=build, pack=pack, stats=stats)
                results.append(row)
                print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

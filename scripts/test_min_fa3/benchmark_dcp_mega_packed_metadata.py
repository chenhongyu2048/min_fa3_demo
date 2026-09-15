"""Compare graph CPU preparation before/after direct C++ queue serialization."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import torch

import dcp_mega_metadata as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    options = parser.parse_args()
    if current._native_packed_queues is None:
        raise RuntimeError("Build the C++ queue module with make first")
    spec = importlib.util.spec_from_file_location("queue_reference", options.reference)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    if reference._native_critical_wave_plan is None:
        raise RuntimeError("The reference must retain the preceding C++ planner optimization")
    requests = json.loads(options.workload.read_text())["measured"]
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    torch.empty(1, dtype=torch.int32, pin_memory=True)
    results = []
    for batch in (4, 16, 64):
        q = tuple(range(batch + 1))
        h = (0, *itertools.accumulate((r["history_tokens"] + 3) // 4 for r in requests[:batch]))
        for scenario, mode, queue_miss, plan_miss in (
            ("native_hit", False, False, False),
            ("native_queue_miss", False, True, False),
            ("auto_hit", None, False, False),
            ("auto_queue_miss", None, True, False),
            ("auto_all_miss", None, True, True),
        ):
            config = dict(hq_local=8, dcp_size=4, num_sms=78, num_comm_sm=8,
                          block_n_override=128, max_num_splits=8, scheduler_heuristic=mode)
            expected = reference.build_dcp_mega_metadata(q, h, **config)
            expected_payload = reference.pack_dcp_mega_metadata(expected, pre_phase=1, post_phase=2)
            target = torch.empty(len(expected_payload), dtype=torch.int32, device="cuda")
            for version, module in (("before", reference), ("after", current)):
                def prepare():
                    if version == "after":
                        return module.build_packed_dcp_mega_metadata(q, h, pre_phase=1, post_phase=2, **config)
                    value = module.build_dcp_mega_metadata(q, h, **config)
                    return value.dispatch, module.pack_dcp_mega_metadata(value, pre_phase=1, post_phase=2)

                module._metadata_queue_cache.clear()
                module._critical_wave_plan_cached.cache_clear()
                _, payload = prepare()
                previous_bytes = payload.tobytes()
                host = torch.frombuffer(payload, dtype=torch.int32).pin_memory()
                samples = []
                for _ in range(options.repeats):
                    if queue_miss:
                        module._metadata_queue_cache.clear()
                        previous_bytes = None
                    if plan_miss:
                        module._critical_wave_plan_cached.cache_clear()
                    info0 = module._critical_wave_plan_cached.cache_info()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    dispatch, payload = prepare()
                    prepared = time.perf_counter()
                    blob = payload.tobytes()
                    if blob != previous_bytes:
                        host = torch.frombuffer(payload, dtype=torch.int32).pin_memory()
                        previous_bytes = blob
                    staged = time.perf_counter()
                    target.copy_(host, non_blocking=True)
                    submitted = time.perf_counter()
                    torch.cuda.synchronize()
                    info1 = module._critical_wave_plan_cached.cache_info()
                    if mode is None:
                        assert info1.misses - info0.misses == int(plan_miss)
                        assert info1.hits - info0.hits == int(not plan_miss)
                    assert vars(dispatch) == vars(expected.dispatch)
                    assert payload == expected_payload
                    torch.testing.assert_close(target.cpu(), torch.frombuffer(expected_payload, dtype=torch.int32), atol=0, rtol=0)
                    samples.append([(prepared-start)*1000, (staged-prepared)*1000,
                                    (submitted-staged)*1000, (submitted-start)*1000])
                row = dict(batch=batch, scenario=scenario, version=version,
                           medians_ms=dict(zip(("build_pack", "pinned_staging", "h2d_submit", "total"),
                                               map(statistics.median, zip(*samples)))),
                           samples_ms=samples)
                results.append(row)
                print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Freeze winning formal cases from three independent uninstrumented Graph runs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

from .config import D1_SELECTION_RULE, write_json


def select_cases(records, cases_per_kind=2):
    groups = defaultdict(dict)
    environment = None
    for record in records:
        if record["schema"] != "motivation.v2.D1":
            raise ValueError("requires motivation v2 D1 records")
        config = record["config"]
        if config["configuration_role"] != "formal" or (config["qhead"], config["kvhead"], config["tp_size"], config["dcp_size"]) != (32, 4, 8, 2):
            raise ValueError("four-GPU smoke data cannot select formal Q32/KV4 cases")
        signature = (record["base_commit"], record["commit"], json.dumps(config, sort_keys=True),
                     record["environment"]["gpu_name"], record["environment"]["sm_count"],
                     record["warmup"], record["iters"])
        if environment is None:
            environment = signature
        elif signature != environment:
            raise ValueError("selection runs must use the same build, GPU and measurement configuration")
        if record["execution_mode"] != "cuda_graph":
            raise ValueError("selection requires CUDA Graph timings")
        for mode in ("critical_wave", "fa3_native"):
            if record["results"][mode]["trace_enabled"]:
                raise ValueError("selection requires uninstrumented timing")
            latency = record["results"][mode]["timing"]["p50_rank_max_ms"]
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("selection requires finite positive measured latency")
        key = (record["workload_kind"], record["case_id"])
        run_id = record["run_id"]
        if run_id in groups[key]:
            raise ValueError(f"duplicate run {run_id} for {key}")
        groups[key][run_id] = record
    selected, candidates = [], []
    run_sets = {tuple(sorted(runs)) for runs in groups.values()}
    if len(run_sets) != 1 or len(next(iter(run_sets), ())) != 3:
        raise ValueError("every candidate needs the same three distinct independent run IDs")
    kinds = ("decode_only_q16", "mixed")
    for kind in kinds:
        winners = []
        for (workload_kind, case_id), runs in groups.items():
            if workload_kind != kind:
                continue
            rows = list(runs.values())
            case = rows[0]["case"]
            if any(row["case"] != case for row in rows):
                raise ValueError("candidate input changed across runs")
            ratios = [r["results"]["fa3_native"]["timing"]["p50_rank_max_ms"] /
                      r["results"]["critical_wave"]["timing"]["p50_rank_max_ms"] for r in rows]
            candidate = {"case_id": case_id, "workload_kind": kind, "speedups": ratios,
                         "median_speedup": median(ratios), "wins_all_runs": all(x > 1 for x in ratios)}
            candidates.append(candidate)
            if candidate["wins_all_runs"]:
                winners.append((candidate, case))
        if len(winners) < cases_per_kind:
            raise ValueError(f"not enough measured critical-wave winners for {kind}; keep selection pending")
        center = median(x[0]["median_speedup"] for x in winners)
        winners.sort(key=lambda x: (abs(x[0]["median_speedup"] - center), int(x[0]["case_id"].rsplit("_", 1)[-1])))
        selected.extend(case | {"selection": stats} for stats, case in winners[:cases_per_kind])
    return {"schema": "motivation.v2.d1_candidates", "selection_status": "measured_formal_winners",
            "selection_rule": D1_SELECTION_RULE,
            "run_ids": list(next(iter(run_sets))), "measurement_signature": environment,
            "all_candidates": candidates, "cases": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
    write_json(args.output_json, select_cases(records))


if __name__ == "__main__":
    main()

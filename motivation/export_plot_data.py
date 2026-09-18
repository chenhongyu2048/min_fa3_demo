"""Print compact plot data using only the Python standard library."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
import sys


DATASETS = ("arxiv", "github", "pile", "freelaw", "prolong")
STRATEGIES = ("all_cp", "br_pbs", "megatron_adapted", "zeppelin_adapted")
COMPONENTS = (
    ("OtherBwd", "others_backward_cuda_critical_rank_avg_ms"),
    ("OtherFwd", "others_forward_cuda_critical_rank_avg_ms"),
    ("CoreFwd", "core_attn_forward_cuda_critical_rank_avg_ms"),
    ("CoreBwd", "core_attn_backward_cuda_critical_rank_avg_ms"),
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_rows(root, experiment):
    paths = sorted((root / "training" / experiment).glob("*.json"))
    if not paths:
        raise ValueError(f"no {experiment} results in {root / 'training'}")
    return sorted((json.loads(path.read_text()) for path in paths),
                  key=lambda record: record["config"]["input_case"]["batch"])


def t3_means(root):
    rows = []
    for dataset in DATASETS:
        groups = defaultdict(list)
        for record in read_jsonl(root / "training" / "T3" / f"{dataset}.jsonl"):
            groups[record["static"]["strategy"]].append(record)
        for strategy in STRATEGIES:
            records = groups[strategy]
            cases = [record["case_id"] for record in records]
            if not cases or len(set(cases)) != len(cases):
                raise ValueError(f"{dataset}/{strategy}: missing or duplicate cases")
            rows.append([dataset, strategy, len(records), *[
                mean(record["timing"][field] for record in records)
                for _, field in COMPONENTS
            ]])
    return rows


def export(root, output):
    writer = csv.writer(output, lineterminator="\n")

    def note(text):
        print(f"# {text}", file=output)

    def table(title, columns, rows):
        note(title)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([format(value, ".9g") if isinstance(value, float) else value
                             for value in row])

    note("PLOT_DATA_V1 BEGIN")
    note(f"run={root.name}; T1/T2/D1 latency: rank-max p50 in ms; T3: mean in ms")
    note("T1 first bar is sum of isolated comm/compute p50s, not p50 of their sum")
    t1 = training_rows(root, "T1")
    t2 = training_rows(root, "T2")
    for label, record in (("T1", t1[0]), ("T2", t2[0])):
        env = record["environment"]
        note(f"{label} gpu={env['gpu_name']}; SMs={env['sm_count']}; "
             f"world={record['config']['world_size']}")
    rows = []
    for record in t1:
        shape = record["config"]["input_case"]
        results = record["results"]
        for method in ("ring", "allgather"):
            comm = results[f"{method}_comm_only"]["p50_rank_max_ms"]
            comp = results[f"{method}_comp_only"]["p50_rank_max_ms"]
            rows.append([shape["batch"], shape["local_seqlen"], method, comm + comp,
                         results[f"{method}_serial"]["p50_rank_max_ms"],
                         results[f"{method}_overlap"]["p50_rank_max_ms"]])
    table("T1", ["batch", "local_seqlen", "method", "comm_plus_compute_ms",
                 "serial_ms", "overlap_ms"], rows)
    profiles = ("step_external_reduce", "step_fused_reduce", "linear_queue_recycle")
    table("T2", ["batch", "local_seqlen", *[name + "_ms" for name in profiles]], [
        [record["config"]["input_case"]["batch"],
         record["config"]["input_case"]["local_seqlen"], *[
             record["results"][name]["timing"]["p50_rank_max_ms"] for name in profiles
         ]] for record in t2
    ])
    note("T3: equal-case arithmetic means; stack the four components to obtain total")
    table("T3", ["dataset", "strategy", "n", *[name + "_ms" for name, _ in COMPONENTS]],
          t3_means(root))

    selected = json.loads((root / "d1_selected.json").read_text())
    records = read_jsonl(root / "d1_selected_trace" / "D1" / "cases.jsonl")
    by_id = {record["case_id"]: record for record in records}
    note("D1: selected rerun only; diagnostic timings are separate from trace-off p50")
    note("trace phase: 0=Q all-gather,1=attention,2=history combine,3=receive,4=final combine")
    note("trace intervals include waits/synchronization; rank/method replays have separate origins")
    for case in selected["cases"]:
        record = by_id[case["case_id"]]
        note("D1_CASE " + json.dumps({
            "case_id": record["case_id"], "kind": record["workload_kind"],
            "config": record["config"], "selection_status": record["selection_status"],
            "gpu": record["environment"]["gpu_name"],
            "sm_count": record["environment"]["sm_count"],
        }, separators=(",", ":")))
        table("D1 trace-off", ["method", "p50_ms"], [
            [method, record["results"][method]["timing"]["p50_rank_max_ms"]]
            for method in ("critical_wave", "fa3_native", "vllm_a2a")
        ])
        diagnostics = record["diagnostics"]
        a2a = diagnostics["vllm_a2a"]
        note(f"A2A critical_rank={a2a['critical_rank']}; CTA traces use rank 0; "
             "CUDA event durations overlap; do not sum all A2A rows")
        table("A2A stages", ["stage", "ms"], a2a["stages_ms"].items())
        for method in ("critical_wave", "fa3_native"):
            trace = diagnostics[method]["trace_by_rank"]["0"]
            origin = min(row[0] for row in trace)
            table(f"TRACE method={method} rank=0; relative integer ns", [
                "cta", "sm", "phase", "start_ns", "duration_ns",
            ], [
                [cta, sm, phase, start - origin, end - start]
                for start, end, cta, sm, phase in trace
            ])
    note("PLOT_DATA_V1 END")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    export(args.run_dir.resolve(), sys.stdout)


if __name__ == "__main__":
    main()

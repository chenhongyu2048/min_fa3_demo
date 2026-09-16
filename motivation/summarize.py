"""Export v2 results without mixing median and mean timing semantics."""
import argparse
import csv
import json
import math
from pathlib import Path


def load_records(run_dir):
    records = []
    for experiment in ("T1", "T2", "T3", "D1"):
        for path in sorted((Path(run_dir) / experiment).glob("*.json*")):
            values = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                      if path.suffix == ".jsonl" else [json.loads(path.read_text())])
            for value in values:
                if value.get("schema") != f"motivation.v2.{experiment}":
                    raise ValueError(f"{path}: requires motivation v2 {experiment}")
                records.append(value)
    if not records:
        raise ValueError("no v2 experiment records found")
    return records


def summary_rows(records):
    rows = []
    for record in records:
        experiment = record["schema"].split(".")[-1]
        case = record.get("case_id", record.get("config", {}).get("case_id"))
        dataset = record.get("dataset", "")
        def add(method, metric, value, statistic):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {experiment}/{case}/{metric}: {value}")
            rows.append(dict(experiment=experiment, dataset=dataset, case_id=case,
                             method=method, metric=metric, value=value, statistic=statistic))
        if experiment in ("T1", "T2"):
            for method, result in record["results"].items():
                timing = result["timing"] if experiment == "T2" else result
                add(method, "latency_ms", timing["p50_rank_max_ms"], "p50_rank_max")
            if experiment == "T1":
                for method, components in record["corun"].items():
                    for name in ("communication_ms", "compute_ms", "total_ms"):
                        add(method, "corun_" + name, components[name]["p50_rank_max_ms"], "p50_rank_max_diagnostic")
        elif experiment == "T3":
            method = record["static"]["strategy"]
            for metric in ("token_imbalance", "attention_imbalance", "communication_tx_bytes", "tile_work", "padding_tokens"):
                add(method, metric, record["static"][metric], "analytical")
            for metric, value in record["timing"].items():
                if metric.endswith("cuda_critical_rank_avg_ms"):
                    add(method, metric, value, "mean_cuda_critical_rank")
        elif experiment == "D1":
            for method, result in record["results"].items():
                if result["trace_enabled"]:
                    raise ValueError("instrumented D1 timings cannot enter performance summary")
                add(method, "graph_latency_ms", result["timing"]["p50_rank_max_ms"], "p50_rank_max_uninstrumented")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summary_rows(load_records(args.run_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

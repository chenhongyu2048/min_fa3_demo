"""Export compact T1 comparisons and representative D1 SM timelines to one JSONL log."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motivation.common import summarize
from motivation.config import PHASES, TRACE_FIELDS
from motivation.trace import decode_trace


def compact_timing(timing, keep_samples=False):
    stats = summarize(timing["per_rank_ms"])
    result = {key: stats[key] for key in ("p50_ms", "p90_ms")}
    if keep_samples:
        result["rank_max_ms"] = stats["rank_max_ms"]
    return result


def select_trace(timing):
    """Choose the replay nearest rank-max p50, then its slowest event-timed rank."""
    stats = summarize(timing["per_rank_ms"])
    sample = min(range(len(stats["rank_max_ms"])),
                 key=lambda index: abs(stats["rank_max_ms"][index] - stats["p50_ms"]))
    rank = max(range(len(timing["per_rank_ms"])),
               key=lambda index: timing["per_rank_ms"][index][sample])
    return sample, rank, stats["rank_max_ms"][sample]


def compact_trace(trace, sm_count):
    decode_trace(trace, sm_count)
    origin = min(row[1] for row in trace[0])
    return {
        "sm_ids_by_cta": [row[0] for row in trace[0]],
        "phase_times_ns": [
            [[stamp - origin for stamp in row[1:5]] for row in phase]
            for phase in trace
        ],
        "rank_sync_ns": [[phase, trace[phase][0][5] - origin,
                          trace[phase][0][6] - origin] for phase in (0, 4)],
    }


def export_plot_data(input_dir, output):
    input_dir, output = Path(input_dir), Path(output)
    rows = [{"type": "schema", "schema": "motivation.v3.plot_data.v1",
             "format": "JSON Lines", "timing_unit": "ms", "trace_unit": "ns",
             "aggregation": "per-sample rank maximum, then p50/p90"}]
    for experiment in ("t1", "d1"):
        path = input_dir / f"{experiment}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        metadata = {key: value for key, value in data.items() if key != "records"}
        rows.append({"type": f"{experiment}_metadata", **metadata})
        if experiment == "t1":
            for record in data["records"]:
                rows.append({"type": "t1_timing",
                             **{key: value for key, value in record.items() if key != "timing"},
                             "timing": compact_timing(record["timing"], keep_samples=True)})
            continue
        rows[-1].update({
            "selection": "sample nearest phased_trace_on rank-max p50, then slowest rank; ties choose lowest index",
            "trace_configuration": "phased_trace_on",
            "time_origin": "earliest phase entry on selected rank; not the CUDA event start",
            "phases": PHASES, "time_fields": TRACE_FIELDS[1:5],
            "phase_times_layout": "[phase_index][cta_index][time_field_index]",
            "rank_sync_fields": ["phase_index", "enter_ns", "exit_ns"],
            "rank_sync_cta": 0,
        })
        for record in data["records"]:
            sample, rank, graph_ms = select_trace(record["timings"]["phased_trace_on"])
            rank_path = input_dir / f'{record["case_id"]}.rank{rank}.json'
            rank_data = json.loads(rank_path.read_text())
            if (rank_data["rank"] != rank or tuple(rank_data["phases"]) != PHASES
                    or tuple(rank_data["trace_fields"]) != TRACE_FIELDS):
                raise ValueError(f"unexpected rank or trace layout in {rank_path}")
            case = record["case"]
            rows.append({
                "type": "d1_timeline", "case_id": record["case_id"],
                "batch_size": len(case["q_lens"]), "total_q": sum(case["q_lens"]),
                "total_history": sum(case["history_lens"]),
                "sample_index": sample, "rank": rank, "dcp_ranks": rank_data["dcp_ranks"],
                "selected_graph_ms": graph_ms,
                "timings": {name: compact_timing(timing)
                            for name, timing in record["timings"].items()},
                **compact_trace(rank_data["sm_trace_samples"][sample],
                                data["environment"]["sm_count"]),
            })
    if len(rows) == 1:
        raise ValueError("input directory contains neither t1.json nor d1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True,
                        help="destination JSON Lines .log file (both options take a file path)")
    args = parser.parse_args(argv)
    rows = export_plot_data(args.input_dir, args.output)
    t1_count = sum(row["type"] == "t1_timing" for row in rows)
    d1_count = sum(row["type"] == "d1_timeline" for row in rows)
    print(f"Wrote {args.output}: {t1_count} T1 configurations, {d1_count} D1 timelines, "
          f"{args.output.stat().st_size} bytes")
    for row in rows:
        if row["type"] == "d1_timeline":
            print(f'{row["case_id"]}: sample_index={row["sample_index"]}, rank={row["rank"]}, '
                  f'DCP={row["dcp_ranks"]}, {len(row["sm_ids_by_cta"])} SMs, '
                  f'graph={row["selected_graph_ms"]:.6f} ms')


if __name__ == "__main__":
    main()

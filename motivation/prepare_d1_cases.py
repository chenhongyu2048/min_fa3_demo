"""Convert existing main trace workloads into unselected D1 candidates."""
import argparse
import json
from pathlib import Path

from .config import D1_SELECTION_RULE, ROOT, write_json


def candidate_manifest(source):
    cases = []
    for line in Path(source).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        phases = set(record["phases"])
        if phases == {"decode"} and all(q == 16 for q in record["q_lens"]):
            kind = "decode_only_q16"
        elif phases == {"decode", "chunk_prefill"}:
            kind = "mixed"
        else:
            continue
        cases.append({"case_id": record["case_id"], "workload_kind": kind,
                      "batch_size": record["batch_size"], "q_lengths": record["q_lens"],
                      "logical_q_lengths": record["logical_q_lens"],
                      "history_lengths": record["history_lens"], "phases": record["phases"]})
    return {"schema": "motivation.v2.d1_candidates", "selection_status": "pending_gpu_measurement",
            "selection_rule": D1_SELECTION_RULE, "source": str(source), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=Path("dcp_test/mega_dcp_trace_cases.jsonl"))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_jsonl if args.source_jsonl.is_absolute() else ROOT / args.source_jsonl
    result = candidate_manifest(source)
    result["source"] = str(args.source_jsonl)
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()

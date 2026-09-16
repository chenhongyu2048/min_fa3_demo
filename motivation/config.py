"""CPU-only experiment definitions; --dry-run never imports torch."""
from pathlib import Path
import json
import subprocess

BASE_COMMIT = "5b0bf71240d4adf0168f9d3d991ff4ed952b08e1"
ROOT = Path(__file__).resolve().parents[1]
TOKENS = 131072
DATASETS = ("arxiv", "github", "pile", "freelaw", "prolong")
STRATEGIES = ("all_cp", "br_pbs", "megatron_adapted", "zeppelin_adapted")
D1_SELECTION_RULE = "win in all three runs; closest to within-kind winners' median speedup; numeric case ID tie-break"


def provenance():
    return {"base_commit": BASE_COMMIT,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))}


def uniform_manifest(gpus):
    if gpus not in (4, 8):
        raise ValueError("gpus must be 4 or 8")
    return {"schema": "motivation.v2.uniform", "world_size": gpus, "cases": [
        {"case_id": f"tokens128k_b{b}", "context": TOKENS, "batch": b,
         "local_seqlen": TOKENS // (b * gpus)} for b in (1, 2, 4, 8, 16)]}


def t3_manifest(seed=0, num_cases=20):
    from balancer import generate_dataset_length_cases
    return {"schema": "motivation.v2.t3_cases", "seed": seed, "target_tokens": TOKENS,
            "num_cases": num_cases, "datasets": {dataset: [
                {"case_id": i, "raw_lengths": lengths, "raw_tokens": sum(lengths)}
                for i, lengths in enumerate(generate_dataset_length_cases(dataset, TOKENS, seed, num_cases))
            ] for dataset in DATASETS}}


def d1_config(gpus):
    if gpus not in (4, 8):
        raise ValueError("gpus must be 4 or 8")
    return {"qhead": 32, "kvhead": gpus // 2, "headdim": 128,
            "tp_size": gpus, "dcp_size": 2,
            "configuration_role": "smoke" if gpus == 4 else "formal"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")

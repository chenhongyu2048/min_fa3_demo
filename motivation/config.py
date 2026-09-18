"""Fixed workloads shared by runners, analysis and CPU checks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOTAL_TOKENS = 131072
BATCH_SIZES = (1, 2, 4, 8, 16)
D1_CASE_IDS = ("case_000003", "case_000009")
PHASES = (
    "q_allgather", "chunk_attention", "history_attention",
    "history_combine_publish", "a2a_pull", "final_combine",
)
TRACE_FIELDS = (
    "sm_id", "entry_ns", "work_start_ns", "work_end_ns",
    "exit_sync_done_ns", "rank_sync_enter_ns", "rank_sync_exit_ns",
)


def t1_cases(world_size=8):
    if world_size < 1 or TOTAL_TOKENS % (2 * max(BATCH_SIZES) * world_size):
        raise ValueError("world_size must give even rank-local sequence lengths")
    return [
        {"batch_size": batch, "total_tokens": TOTAL_TOKENS,
         "global_seqlen": TOTAL_TOKENS // batch,
         "local_seqlen": TOTAL_TOKENS // (batch * world_size)}
        for batch in BATCH_SIZES
    ]


def d1_cases():
    source = ROOT / "dcp_test" / "mega_dcp_trace_cases.jsonl"
    records = {row["case_id"]: row for row in
               (json.loads(line) for line in source.read_text().splitlines() if line.strip())}
    return [records[case_id] for case_id in D1_CASE_IDS]


def d1_topology(world_size, smoke=False):
    expected_world = 4 if smoke else 8
    if world_size != expected_world:
        raise ValueError(f"D1 requires {expected_world} GPUs for {'smoke' if smoke else 'formal'} execution")
    # Preserve Hq_local=4 and DCP=2 so smoke uses the same kernel specialization.
    return {"tp_size": world_size, "dcp_size": 2, "q_heads": world_size * 4,
            "kv_heads": world_size // 2, "configuration_role": "smoke" if smoke else "formal"}


def add_sampling_args(parser):
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)


def validate_sampling(args):
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")

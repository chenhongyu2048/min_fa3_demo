"""Focused CPU checks, including actual baseline control flow with fake CUDA ops."""

import ast
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from motivation.common import measure, summarize
from motivation.config import ROOT, d1_cases, d1_topology, t1_cases
from motivation.trace import decode_trace


def load_functions(path, names, namespace, class_name=None):
    """Execute the real function bodies without importing unavailable torch/CUDA.

    These tests exercise scheduling, not tensor arithmetic or GPU correctness.
    """
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Tensor:
    dtype = "bf16"

    def __init__(self, name="tensor"):
        self.name = name

    def size(self, dim):
        return 3

    def contiguous(self):
        return self

    def to(self, *args):
        return self

    def __getitem__(self, key):
        return Tensor(f"{self.name}[{key}]")

    def __setitem__(self, key, value):
        pass

    def __floordiv__(self, value):
        return self


def example_trace():
    return [
        [[7, 100 + phase * 100, 120 + phase * 100, 150 + phase * 100, 190 + phase * 100,
          105 + phase * 100 if phase in (0, 4) else 0,
          115 + phase * 100 if phase in (0, 4) else 0],
         [2, 101 + phase * 100, 122 + phase * 100, 180 + phase * 100, 191 + phase * 100, 0, 0]]
        for phase in range(6)
    ]


class WorkloadTest(unittest.TestCase):
    def test_smoke_topology_preserves_kernel_specialization(self):
        for world, smoke in ((4, True), (8, False)):
            config = d1_topology(world, smoke)
            self.assertEqual(config["q_heads"] // world, 4)
            self.assertEqual(config["dcp_size"], 2)
            self.assertEqual(config["kv_heads"], world // 2)
        with self.assertRaises(ValueError):
            d1_topology(4)

    def test_t1_fixed_tokens(self):
        cases = t1_cases()
        self.assertEqual([case["batch_size"] for case in cases], [1, 2, 4, 8, 16])
        self.assertEqual([case["local_seqlen"] for case in cases], [16384, 8192, 4096, 2048, 1024])
        for world in (2, 4, 8):
            for case in t1_cases(world):
                self.assertEqual(case["batch_size"] * case["local_seqlen"] * world, 131072)
        with self.assertRaises(ValueError):
            t1_cases(3)

    def test_d1_cases_and_native_descriptor_domains(self):
        from dcp_mega_metadata import build_dcp_mega_metadata, pack_dcp_mega_metadata

        cases = d1_cases()
        self.assertEqual([case["case_id"] for case in cases], ["case_000003", "case_000009"])
        self.assertEqual(cases[0]["q_lens"], [16] * 42)
        self.assertEqual(cases[1]["q_lens"], [16] * 29 + [4096, 4096])
        self.assertEqual([sum(case["history_lens"]) for case in cases], [404697, 355534])
        for case in cases:
            for rank in range(2):
                cu_q = list(itertools.accumulate(case["q_lens"], initial=0))
                local_history = [(length + 1 - rank) // 2 for length in case["history_lens"]]
                cu_k = list(itertools.accumulate(local_history, initial=0))
                metadata = build_dcp_mega_metadata(
                    cu_q, cu_k, hq_local=4, dcp_size=2, num_sms=132, num_comm_sm=4,
                    scheduler_heuristic=False, reorder_history_override=False)
                self.assertEqual(metadata.split_policy, "fa3_native")
                self.assertEqual(metadata.history_order_policy, "fifo")
                image = pack_dcp_mega_metadata(metadata, pre_phase=1, post_phase=2)
                chunk_count, history_count = image[8:10]
                self.assertEqual(chunk_count + history_count, len(metadata.attention))
                self.assertTrue(all(row[0] == 0 for row in metadata.attention[:chunk_count]))
                self.assertTrue(all(row[0] == 1 for row in metadata.attention[chunk_count:]))
                ids = [row[7] for row in metadata.attention]
                self.assertEqual(sorted(ids), list(range(len(ids))))
                self.assertTrue(set(metadata.publish_dependencies).issubset(ids[chunk_count:]))
                self.assertTrue(set(metadata.final_dependencies).issubset(ids[:chunk_count]))


class TimingTest(unittest.TestCase):
    def test_reduce_each_sample_before_quantiles(self):
        summary = summarize([[1, 9, 2], [8, 1, 10]])
        self.assertEqual(summary["rank_max_ms"], [8, 9, 10])
        self.assertEqual(summary["p50_ms"], 9)
        self.assertEqual(summary["p90_ms"], 9.8)
        with self.assertRaises(ValueError):
            summarize([[1], [1, 2]])

    def test_every_sample_excludes_world_barrier_and_observation(self):
        log = []
        torch = types.ModuleType("torch")
        dist = types.ModuleType("torch.distributed")
        torch.distributed = dist
        names = iter(("start", "end"))

        def event(**kwargs):
            name = next(names)
            return types.SimpleNamespace(
                record=lambda: log.append(name), synchronize=lambda: log.append("end_sync"),
                elapsed_time=lambda end: 2.0)
        torch.cuda = types.SimpleNamespace(Event=event, synchronize=lambda device: log.append("device_sync"))
        dist.barrier = lambda **kwargs: log.append("world_barrier")
        dist.get_world_size = lambda: 1

        def gather(destination, value):
            log.append("gather_results")
            destination[0] = value
        dist.all_gather_object = gather
        with mock.patch.dict("sys.modules", {"torch": torch, "torch.distributed": dist}):
            timing, observed = measure(
                lambda: log.append("replay"), types.SimpleNamespace(index=0), 1, 2,
                observe=lambda: log.append("observe"))
        sample = ["device_sync", "world_barrier", "device_sync", "start", "replay",
                  "end", "end_sync", "observe"]
        self.assertEqual(log, ["start", "end", "replay"] + sample * 2 + ["gather_results"])
        self.assertEqual(timing["rank_max_ms"], [2.0, 2.0])
        self.assertEqual(len(observed), 2)


class BaselineControlTest(unittest.TestCase):
    def test_allgather_modes_use_existing_compute_and_order_functions(self):
        namespace = load_functions("ring_test/hybrid_forward_baselines.py", {"forward"}, {},
                                   "VarlenAllGatherForward")
        for mode in ("comm_only", "comp_only", "serial", "overlap"):
            log = []
            fake = types.SimpleNamespace(k=Tensor(), q=Tensor(), out=Tensor(), heads_k_stride=1)
            fake._start_kv_all_gather = lambda buffer, head: log.append(("send", head)) or head
            fake._wait_kv_all_gather = lambda work: log.append(("wait", work))
            fake._order_kv_chunk = lambda buffer, data: log.append(("order", data))
            fake._q_head_slice = lambda head: head
            fake.q_chunk = types.SimpleNamespace(copy_=lambda q: None)
            fake._compute_chunk = lambda buffer, head: log.append(("compute", head))
            result = namespace["forward"](fake, execution_mode=mode, gathered_kv=["kv0", "kv1", "kv2"])
            sends = [value for event, value in log if event == "send"]
            computes = [value for event, value in log if event == "compute"]
            self.assertEqual(sends, [] if mode == "comp_only" else [0, 1, 2])
            self.assertEqual(computes, [] if mode == "comm_only" else [0, 1, 2])
            if mode == "comm_only":
                self.assertIsNone(result)
                self.assertFalse(any(event == "order" for event, _ in log))
            if mode == "comp_only":
                self.assertEqual([value for event, value in log if event == "order"], ["kv0", "kv1", "kv2"])
            if mode == "serial":
                self.assertLess(log.index(("compute", 0)), log.index(("send", 1)))
            if mode == "overlap":
                self.assertLess(log.index(("send", 1)), log.index(("compute", 0)))

    def test_ring_modes_and_causal_slices(self):
        for causal in (False, True):
            for mode in ("comm_only", "comp_only", "serial", "overlap"):
                with self.subTest(causal=causal, mode=mode):
                    log, pending = [], [False]

                    class Comm:
                        rank, world_size = 1, 3

                        def __init__(self, *args):
                            self.step = 0

                        def send_recv_kv(self, k, v):
                            self.step += 1
                            self.assert_not_pending()
                            pending[0] = True
                            log.append("send")
                            return Tensor(f"remote{self.step}"), Tensor()

                        def assert_not_pending(self):
                            if pending[0]:
                                raise AssertionError("send before previous transfer was joined")

                        def wait(self):
                            if not pending[0]:
                                raise AssertionError("wait without send")
                            pending[0] = False
                            log.append("wait")

                    def attention(q, k, v, *args):
                        if mode == "serial":
                            self.assertFalse(pending[0])
                        log.append(("compute", k.name))
                        return Tensor("out"), Tensor("lse")

                    ns = {"RingComm": Comm, "get_half_index": lambda cu, front: "front" if front else "back",
                          "update_out_and_lse": lambda out, lse, block, block_lse: (block, block_lse)}
                    load_functions("ring_test/ring_common.py",
                                   {"ring_varlen_forward", "zigzag_ring_varlen_forward"}, ns)
                    kv = [(Tensor(f"cached{step}"), Tensor()) for step in range(3)]
                    common = dict(execution_mode=mode, gathered_kv=kv)
                    if causal:
                        result = ns["zigzag_ring_varlen_forward"](
                            None, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), 8, attention, **common)
                    else:
                        result = ns["ring_varlen_forward"](
                            None, Tensor(), Tensor(), Tensor(), False, attention, **common)
                    computes = [entry for entry in log if isinstance(entry, tuple)]
                    self.assertEqual(log.count("send"), 0 if mode == "comp_only" else 2)
                    self.assertEqual(log.count("wait"), 0 if mode == "comp_only" else 2)
                    self.assertEqual(len(computes), 0 if mode == "comm_only" else 3)
                    if mode == "comm_only":
                        self.assertIsNone(result)
                    if mode == "comp_only":
                        for step, entry in enumerate(computes):
                            self.assertTrue(entry[1].startswith(f"cached{step}"))
                    if mode == "overlap":
                        self.assertLess(log.index(computes[0]), log.index("wait"))


class TraceTest(unittest.TestCase):
    def test_waits_and_noncontiguous_physical_sms(self):
        decoded = decode_trace(example_trace(), 2)
        self.assertEqual(decoded[0]["rank_wait_ns"], 10)
        self.assertEqual(decoded[1]["rank_wait_ns"], 0)
        self.assertEqual(decoded[0]["tail_idle_sm_ns"], 30)
        self.assertEqual(decoded[0]["work_span_ns"], 60)
        self.assertEqual(decoded[0]["ctas"][0]["exit_wait_ns"], 40)

    def test_invalid_coverage_and_barrier_order(self):
        for change in (lambda t: t[0][1].__setitem__(0, 7),
                       lambda t: t[0][0].__setitem__(4, 160),
                       lambda t: t[0][0].__setitem__(6, 130),
                       lambda t: t[1][0].__setitem__(1, 180)):
            trace = example_trace()
            change(trace)
            with self.assertRaises(ValueError):
                decode_trace(trace, 2)

    def test_cpu_analysis_export(self):
        from motivation.analyze import analyze

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing = summarize([[1.0], [2.0]])
            data = {"environment": {"world_size": 2, "sm_count": 2}, "records": [
                {"case_id": "example", "timings": {key: timing for key in
                 ("mixed_trace_off", "phased_trace_off", "phased_trace_on", "vllm_a2a_events_off", "vllm_a2a_events_on")}}]}
            (root / "d1.json").write_text(json.dumps(data))
            for rank in range(2):
                (root / f"example.rank{rank}.json").write_text(json.dumps({
                    "sm_trace_samples": [example_trace()], "vllm_phase_samples_ms": [{"history_ms": rank + 1.0}]}))
            analyze(root, root / "tables", plots=False)
            self.assertEqual(len((root / "tables" / "d1_stages.csv").read_text().splitlines()), 13)
            overhead = json.loads((root / "tables" / "d1_overhead.json").read_text())
            self.assertEqual(overhead[0]["trace_delta_percent"], 0)


if __name__ == "__main__":
    unittest.main()

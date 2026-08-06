# Mega DCP Trace Workload Research and Implementation Log

Last updated: 2026-08-06 10:41 CST

This file is the persistent record for the trace-workload extractor. Research
facts, design choices, implementation batches, and every test invocation are
recorded here as the work happens. Upstream facts and locally derived results
are deliberately labelled separately.

## 1. Goal

Build a deterministic, CPU-only replay tool that converts a real Mooncake/Kimi
conversation trace into random wall-clock snapshots suitable for Mega DCP
varlen kernel inputs:

- request count (`batch_size`);
- current query/chunk lengths (`q_lens`);
- history KV lengths (`history_lens`);
- total KV lengths (`total_kv_lens = history_lens + q_lens`);
- aligned request and phase metadata for diagnosis.

The extractor, documentation, tests, and example configuration live entirely
under `dcp_test/trace/`. It does not modify or invoke the CUDA kernel.

## 2. Public Research

Sources were inspected before implementation. Access date: 2026-08-05.

| Work | Relevance | Source |
| --- | --- | --- |
| Mooncake FAST'25 | Disaggregated serving system and Kimi production context | https://www.usenix.org/conference/fast25/presentation/qin |
| Mooncake FAST'25 release | Public conversation trace used by this tool | https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release |
| vLLM MLA DCP | Decode context parallel implementation | https://github.com/vllm-project/vllm/pull/23734 |
| vLLM GQA DCP | GQA decode context parallel implementation | https://github.com/vllm-project/vllm/pull/24864 |
| vLLM packed A2A | More recent DCP communication work | https://github.com/vllm-project/vllm/pull/41160 |
| Sarathi-Serve | Chunked-prefill scheduling and decode/prefill coexistence | https://arxiv.org/abs/2403.02310 |
| LoongServe | Elastic sequence parallelism for long-context serving | https://arxiv.org/abs/2404.09526 |
| Infinite-LLM | Distributed long-context serving | https://arxiv.org/abs/2401.02669 |
| Vidur | Trace-driven LLM serving simulation | https://github.com/microsoft/vidur |
| Azure LLM 2024 | Alternative public inference workload dataset | https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md |
| BurstGPT v2.0 | Alternative workload with session IDs | https://github.com/HPMLL/BurstGPT/releases/tag/v2.0 |

### 2.1 Mooncake trace facts

Upstream file:
`FAST25-release/traces/conversation_trace.jsonl`

- Git blob SHA: `5a371794b81d7fc7487ec8a02a2c631bcc7fbdf8`
- File SHA-256: `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df`
- Raw fields: `timestamp`, `input_length`, `output_length`, `hash_ids`
- `timestamp` is elapsed milliseconds; the final value is 3,536,999, which
  corresponds to the measured duration of about 3,537 seconds.
- Every inspected record satisfies
  `len(hash_ids) == ceil(input_length / 512)`.

The following are local measurements of that exact file, not fields published
in each trace row:

- 12,031 requests over about 3,537 seconds, about 3.40 requests/second.
- Input length mean 12,035; P50/P90/P99 6,909/27,367/85,400; max 126,195.
- Output length mean 342.62; P50/P90/P99 350/597/1,118.5; max 2,000.
- Timestamps are usually aggregated in roughly three-second buckets.
- Requests per timestamp P50 10, P90 15, max 28.
- With an infinite cache and only prior, consecutive, complete 512-token
  blocks counted, the locally derived token hit ratio is about 37.34%.

The last number is an analysis result, not a production cache-hit metric. It
does not include finite capacity, routing, active-block pinning, or output KV.

### 2.2 BurstGPT limitation

Only `BurstGPT_3.csv` in v2.0 contains both `Session ID` and `Elapsed time`.
Within a session, `Request tokens` can decrease, which is consistent with
truncation, editing, or model changes. Therefore the implementation does not
infer a monotonic conversation state from BurstGPT and does not combine its
distribution with Mooncake. BurstGPT is retained as supporting evidence that
session IDs alone are insufficient for reconstructing KV state.

## 3. Design Decisions

### 3.1 Trace and time

- Verify the configured SHA-256 before parsing.
- Assign stable request IDs from zero-based JSONL line numbers.
- Reject malformed rows and drop, with counters, requests whose
  `input_length + output_length` exceeds `max_model_len`.
- `preserve` keeps timestamp buckets intact.
- `uniform_bucket_jitter` deterministically spreads tied requests between the
  current and next distinct timestamp; the final bucket uses the median
  positive bucket width.
- Convert source milliseconds to integer microseconds with decimal arithmetic.
- Scale elapsed arrival time as `elapsed / arrival_time_scale`, so a scale
  greater than one raises offered load.

### 3.2 Prefix cache

- Model a finite global LRU of full 512-token prompt blocks.
- Accept only a consecutive hit from the first prompt block.
- Leave at least one prompt token uncached so a query always produces logits.
- Insert/touch a prompt block only after replay has computed the full block.
- Never invent hashes for output tokens or infer reuse from request adjacency.
- Cache capacity describes reusable prefix blocks, not active-request KV
  memory or request admission capacity.

### 3.3 Scheduler and MTP

- Replay one logical DCP instance at fixed `fixed_step_us` intervals.
- Admit arrivals FCFS up to `max_num_seqs` active requests.
- Schedule decode/MTP before chunked prefill; retain FCFS within each phase.
- Schedule at most one query per request per step.
- Prefill may be shortened by `prefill_chunk_size` and remaining token budget.
- MTP queries are atomic and are deferred if they do not fit the remaining
  `max_num_batched_tokens` budget.
- Default `target_plus_drafts` schedules
  `q_len = 1 + num_speculative_tokens`; `drafts_only` is available explicitly.
- `accepted_draft_pmf` chooses committed draft tokens. At the output-length
  boundary, accepted drafts are capped but the scheduled query remains intact.
- Arrival jitter, draft acceptance, and reservoir sampling use independent RNG
  streams derived from the configured seed.

### 3.4 Mega DCP cases

- Record the scheduled state before applying the step's state transition.
- Simulate every scheduled request, but include only requests with
  `history_len >= dcp_size` in the Mega workload. This guarantees non-empty
  token-interleaved local history on every DCP rank.
- A mixed scheduler step can therefore produce a smaller Mega batch while its
  noneligible work still consumes the global token budget.
- Uniformly sample eligible fixed-width wall-clock steps without replacement
  using a one-pass reservoir.
- Empty/noneligible steps are counted but do not produce `B=0` cases.

## 4. Output Contract

The CLI is:

```bash
python -m dcp_test.trace.generate --config CONFIG.json --output CASES.jsonl
```

Existing outputs are protected unless `--force` is passed. Relative trace
paths are resolved relative to the config file. A case contains provenance,
sample time and scheduler step, the four kernel input dimensions, and aligned
per-request phase/debug arrays. Output is sorted by sampled time and is
byte-for-byte deterministic for an identical trace, configuration, and seed.

The example configuration is explicitly illustrative. Scheduler limits,
cache capacity, fixed step, arrival scaling, and MTP acceptance are not present
in the public trace, so the tool never labels chosen values as Kimi production
defaults.

## 5. Implementation Log

### 2026-08-05 23:24 CST - Repository and input inspection

- Confirmed the git worktree was clean before edits.
- Confirmed `dcp_test/trace/` did not previously exist.
- Confirmed `dcp_test/benchmark_dcp_varlen.py` consumes aligned query and
  history lengths and rejects empty rank-local history.
- Located `/tmp/mooncake_conversation_trace_full.jsonl` outside the repository.
- Recomputed its SHA-256 successfully and inspected the actual row schema.
- Created this worklog before implementation code, as required.

### 2026-08-05 23:30 CST - Core implementation batch

- Added strict JSON configuration loading and canonical `config_sha256`.
- Added independent deterministic RNG streams for arrival jitter, MTP draft
  acceptance, and reservoir sampling.
- Added SHA-first Mooncake parsing, source-order validation, integer-microsecond
  arrival normalization, model-length filtering, and both timestamp policies.
- Added a finite complete-block LRU with consecutive-prefix matching and the
  required one-token cache-hit guard.
- Added the fixed-step, decode-first scheduler. Noneligible work is applied to
  scheduler state and budget even when it is absent from the Mega snapshot.
- Added output-length-aware MTP commits, random eligible-step reservoir
  sampling, aligned JSON case construction, and replay counters.
- Added a CLI that refuses accidental overwrite and writes JSONL atomically.
- Clarified the alternate `drafts_only` rule in code: its configured count is
  total query length, so its PMF has entries for accepted counts `0..q_len-1`.
  The default and example remain `target_plus_drafts`.
- No tests were run during this batch.

### 2026-08-05 23:32 CST - Tests and example configuration

- Added one consolidated CPU test module with eight test methods. Subcases
  cover strict configuration validation, trace provenance and schema, both
  timestamp modes, model-length filtering, prefix/LRU behavior, scheduler
  state transitions, MTP atomicity, mixed phases, deterministic reservoir
  sampling, insufficient-case errors, and CLI overwrite protection.
- Added an example configuration using the verified Mooncake SHA and 64 output
  cases. Its cache, scheduler, arrival, and acceptance values are illustrative
  research inputs, not claimed Kimi production settings.
- The example expects `conversation_trace.jsonl` beside the config; the large
  upstream file is intentionally not copied into the repository.
- No tests were run during this batch.

## 6. Test Run Log

Planned process-level test invocations are intentionally limited to three:

1. One syntax/static pass with `python -m compileall -q dcp_test/trace`.
2. One consolidated CPU suite with
   `python -m unittest discover -s dcp_test/trace/tests -v`.
3. One full Mooncake trace generation smoke test to a temporary output path.

Any failure-driven rerun will be recorded here with its reason. Multiple seeds
and configuration variants are exercised inside the consolidated process, not
as separate test commands.

### Invocation 1 - static syntax check

- Time: 2026-08-05 23:34 CST
- Command: `python -m compileall -q dcp_test/trace`
- Result: PASS (exit code 0, no diagnostics)
- Scope: all Python implementation and test files under `dcp_test/trace/`

Total test/check process invocations so far: 1. No reruns.

### Invocation 2 - consolidated CPU suite

- Time: 2026-08-05 23:35 CST
- Command: `python -m unittest discover -s dcp_test/trace/tests -v`
- Result: PASS (8 tests, 0 failures, 0 errors, about 0.010 seconds)
- Test runner processes: 1
- Covered internal scenarios: strict config and two query rules; trace SHA,
  schema, timestamp, and length filtering; prefix/LRU behavior; chunk-to-decode
  transitions; MTP budget atomicity; mixed-phase cases; real prefix reuse;
  same-seed determinism; alternate sampling seed; sampling shortfall; atomic
  output; overwrite refusal and forced deterministic replacement.

Total test/check process invocations so far: 2. No reruns.

### Invocation 3 - first full Mooncake smoke (semantic failure)

- Time: 2026-08-05 23:35 CST
- Command: `python -m dcp_test.trace.generate --config
  /tmp/mega_dcp_smoke_config_20260805.json --output
  /tmp/mega_dcp_mooncake_smoke_20260805_2335.jsonl`
- Process result: exit code 0 and 64 syntactically valid JSONL rows.
- Acceptance result: FAIL. Replay admitted only 15 of 12,031 loaded requests,
  processed 2,246 scheduler steps, and counted 3,599,842 idle steps.
- Root cause: the public trace's `timestamp` is milliseconds, but the first
  parser version treated it as seconds. The trace tail value 3,536,999 proves
  the unit when compared with the measured duration of about 3,537 seconds.
- Action: changed Mooncake conversion from `timestamp * 1_000_000` to
  `timestamp * 1_000` and updated the timestamp unit test expectation.
- The generated `/tmp/mega_dcp_mooncake_smoke_20260805_2335.jsonl` is invalid
  for workload use and will be replaced by the corrected smoke run.

Total test/check process invocations so far: 3. One required correction; no
duplicate or discretionary reruns.

### Invocation 4 - post-fix static syntax check

- Time: 2026-08-05 23:36 CST
- Command: `python -m compileall -q dcp_test/trace`
- Result: PASS (exit code 0, no diagnostics)
- Reason for rerun: Python timestamp conversion and its unit test changed.

Total test/check process invocations so far: 4. One failure-driven static
rerun; no unrelated commands.

### Invocation 5 - post-fix consolidated CPU suite

- Time: 2026-08-05 23:37 CST
- Command: `python -m unittest discover -s dcp_test/trace/tests -v`
- Result: PASS (8 tests, 0 failures, 0 errors, about 0.011 seconds)
- Reason for rerun: the timestamp conversion assertion changed from a
  one-second interpretation to the trace's actual one-millisecond interval.

Total test/check process invocations so far: 5. Two correction-driven reruns;
no extra seed/config process launches.

### Invocation 6 - corrected full Mooncake smoke

- Time: 2026-08-05 23:37 CST
- Command: `python -m dcp_test.trace.generate --config
  /tmp/mega_dcp_smoke_config_20260805.json --output
  /tmp/mega_dcp_mooncake_smoke_20260805_2335.jsonl --force`
- Result: PASS (exit code 0, about 4.53 seconds, 64 cases written)
- Input coverage: 12,031/12,031 requests admitted and completed; zero requests
  dropped by the illustrative 131,072-token model limit.
- Replay totals: 1,240,339 processed steps; 1,240,338 eligible steps in the
  sampling window; 1,530,391 scheduled queries; 1,530,390 eligible queries;
  1 query filtered for insufficient rank-local history.
- Phase totals: 1,499,428 decode/MTP queries and 30,963 chunk-prefill queries.
- Prefix and concurrency totals: 50,181,632 cached-prefix tokens credited;
  maximum 6 active requests under this illustrative arrival/scheduler profile.
- The earlier invalid temporary JSONL was atomically replaced by this corrected
  output. The large source trace remains outside the repository.

Final planned test/check process count: 6. This comprises the original three
commands plus the minimum three post-fix repeats (static, CPU, real trace).
There were no discretionary repeats or per-seed process launches.

### 2026-08-05 23:39 CST - Corrected workload inspection

- A read-only `jq` aggregation was attempted after testing, but `jq` is not
  installed (`exit code 127`). It performed no analysis and changed no files.
- The fallback used Python's structured JSON parser once; this was artifact
  analysis, not another implementation test or replay.
- Across the 64 sampled cases, batch size min/P50/P90/max is 1/1/2/3.
- Across 76 aligned queries, q length min/P50/P90/max is 5/5/5/4,096.
- History length min/P50/P90/P99/max is
  512/5,073/25,446/40,215/49,664.
- Phase counts are 72 decode/MTP and 4 chunk-prefill queries. One case contains
  both phases.
- These numbers describe only `example_config.json`'s illustrative replay
  settings. They must not be presented as Kimi production scheduler metrics.

### 2026-08-05 23:40 CST - Final repository review

- Confirmed the worktree contains only the nine intended new files under
  `dcp_test/trace/`; no existing kernel, benchmark, or generated artifact was
  changed.
- Inspected the corrected output's first and last cases. Their sampled times
  are 43.448 seconds and 3,526.742 seconds, confirming coverage across the
  trace rather than only its first timestamp buckets.
- Code did not change after the passing post-fix static, CPU, and real-trace
  checks. This final Markdown-only entry does not require another test rerun.

## 7. Known Modeling Limits

- Fixed steps are a scheduling abstraction, not measured kernel duration.
- No multi-instance routing, preemption, KV swap, network transfer, or active
  KV memory pressure is modeled.
- Public Mooncake hashes establish prompt-block identity but provide no hash
  identity for generated tokens.
- Random samples are conditional on a Mega-eligible launch; they are not
  unconditional samples of idle wall-clock time.

## 8. Example Viewer

### 2026-08-06 10:38 CST - Request and interface design

- Requested a reusable replacement for the ad hoc Python command previously
  used to display a few randomly selected generated cases.
- The viewer will read an existing workload JSONL and will not replay the trace
  or modify the workload file.
- Planned module entry point: `python -m dcp_test.trace.show_example`.
- Required interface: `--input`; optional `--num-examples` (default 5),
  `--seed` (default 20260806), and `--full`.
- Default output will retain provenance identifiers and the aligned scheduler,
  kernel-shape, cache, generation, and MTP fields needed to understand each
  selected case. `--full` will print each complete source JSON object.
- Selection is without replacement and deterministic for the same file, count,
  and display seed. Requesting more examples than available will display all
  available cases rather than fail.
- No implementation test has been run for this addition yet.

### 2026-08-06 10:39 CST - Implementation and usage

- Added `show_example.py`, using only the Python standard library.
- The loader rejects unreadable, empty, malformed, non-object, or incompatible
  JSONL input with a line-specific error and exit code 2.
- The default projection contains `case_id`, trace/config provenance, sample
  time and step, batch/request IDs, phases, q/history/total-KV lengths,
  prompt/output/cache/generation state, MTP acceptance, and replay parameters.
- The display seed controls only which existing cases are printed. It is
  independent of the replay seed stored in each case and never changes the
  generated workload.

First generate a workload from a valid replay configuration:

```bash
python -m dcp_test.trace.generate \
  --config dcp_test/trace/example_config.json \
  --output /tmp/mooncake_cases_seed42.jsonl
```

The current edited example uses `target_plus_drafts` with 16 speculative
tokens. Before generation, its `accepted_draft_pmf` must contain 17 entries for
accepted counts `0..16`; at the time of this log entry it still contains the
old five-entry PMF. This viewer does not modify or reinterpret that config.

Display five focused examples with the default deterministic display seed:

```bash
python -m dcp_test.trace.show_example \
  --input /tmp/mooncake_cases_seed42.jsonl
```

Choose a count and display seed explicitly:

```bash
python -m dcp_test.trace.show_example \
  --input /tmp/mooncake_cases_seed42.jsonl \
  --num-examples 8 \
  --seed 1234
```

Print complete source cases rather than the focused field subset:

```bash
python -m dcp_test.trace.show_example \
  --input /tmp/mooncake_cases_seed42.jsonl \
  --num-examples 3 \
  --seed 1234 \
  --full
```

Selected cases are written as JSONL to stdout. A one-line summary containing
the input path, total case count, displayed count, display seed, and full-mode
flag is written to stderr. The same input/count/seed produces the same selected
cases. Selection is without replacement; a count larger than the input corpus
prints every case once.

- Added one in-process unit test for focused projection, deterministic
  selection, uniqueness, full output, and oversized-count capping.
- No test command had been run before the implementation batch completed.

### Example viewer validation 1/3 - static syntax

- Time: 2026-08-06 10:40 CST
- Command: `python -m compileall -q dcp_test/trace`
- Result: PASS (exit code 0, no diagnostics)
- Process-level validation count for this addition: 1; no reruns.

### Example viewer validation 2/3 - consolidated CPU suite

- Time: 2026-08-06 10:40 CST
- Command: `python -m unittest discover -s dcp_test/trace/tests -v`
- Result: PASS (9 tests, 0 failures, 0 errors, about 0.011 seconds)
- The ninth method is the new viewer test; all eight replay/generator tests
  continue to pass in the same process.
- Process-level validation count for this addition: 2; no reruns.

### Example viewer validation 3/3 - real workload CLI smoke

- Time: 2026-08-06 10:41 CST
- Command: `python -m dcp_test.trace.show_example --input
  /tmp/mega_dcp_mooncake_smoke_20260805_2335.jsonl --num-examples 3 --seed
  1234`
- Result: PASS (exit code 0).
- Input contained 64 real Mooncake-derived cases. The command printed three
  focused JSON objects for `case_000056`, `case_000014`, and `case_000000`,
  followed on stderr by a summary reporting `examples_displayed=3`,
  `display_seed=1234`, and `full=false`.
- Final process-level validation count for this addition: 3; no failures or
  reruns. No trace replay, CUDA build, or GPU test was needed.
- Only this Markdown record changed after the passing validations.

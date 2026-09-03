# Inference integrations

Inference-framework-specific code is grouped here and kept separate from the
standalone min-FA3 and training/Transformer-layer benchmark paths.

- `vllm_bench/` contains the workload converter, service launcher, TBT client,
  matrix runner, synthetic Llama configurations, and tests. Its import name
  remains `vllm_bench`; repository scripts add `infer/` to `PYTHONPATH`.
- `vllm_plugin/` is the editable `min-fa3-vllm-plugin` package registered
  through vLLM's `vllm.general_plugins` entry point.

See [`vllm_bench/README.md`](vllm_bench/README.md) for installation, smoke, and
formal benchmark commands.

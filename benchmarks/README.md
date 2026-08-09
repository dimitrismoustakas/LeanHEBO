# Benchmark policy

LeanHEBO performance claims require a baseline pinned to Huawei HEBO commit
`ee6112d39d1a9e9703fecaf9057193e1ec9dae72` and matched algorithmic work.

Measure initial Sobol suggestions, the first model-based suggestion, steady-state suggestions,
observation, and checkpoint/resume independently. Record all configuration fields, Torch and
GPyTorch versions, device, dtype, thread counts, hardware, posterior calls, GP steps, jitter,
RSS, and CUDA memory where applicable.

The end-to-end matrix covers 16–1,024 observations, dimensions 5/20/50, continuous and mixed
spaces, batches 1/4/8/32, both dtypes, declared thread counts, and CUDA where available. CPU and
CUDA timings include synchronization and transfer cost. GPU acceleration is reported only for
repeatable cells with at least a 1.5× median improvement and passing quality tests.

Changed training steps, refit schedules, populations, or generations are changed-work results,
not equivalent-work speedups. Optimization quality uses matched evaluation budgets and multiple
seeds, reporting normalized regret, uncertainty intervals, failures, and invalid suggestions.

Structural performance properties—posterior call count, persistent GP construction, absence of
hot-path table/NumPy conversions, and cache invalidation—belong in ordinary CI. Fragile wall-time
regressions belong on dedicated benchmark hardware.

## Local foundation

The raw result contract is versioned in `schema/result.schema.json`. The helper types in
`harness/` record wall and process CPU samples and reject comparisons unless every declared work
field matches. Result writers refuse to overwrite an existing raw record.

Prepare the audited upstream source and its dependencies in an isolated, ignored environment:

```bash
uv run python benchmarks/environments/prepare_upstream.py
```

Use `--checkout-only` to validate the exact commit and pinned setup-file digests without resolving
the old upstream dependency set. Upstream HEBO and Pymoo are development/reference inputs only;
neither is imported by `src/leanhebo`.

Run the structural search benchmark and focused harness tests with:

```bash
uv run pytest benchmarks/micro/benchmark_matched_search.py
uv run pytest benchmarks/tests
```

The toy suite defaults to one seed and intentionally small work. It proves that the runner works;
it is not evidence for a speed or quality claim:

```bash
uv run python -m benchmarks.quality.run_toy_suite --seeds 0
uv run python -m benchmarks.profiling.run_cprofile --seed 0
```

For a real comparison, run LeanHEBO and upstream in their respective environments, predeclare the
multi-seed protocol, and compare only records whose complete `work` objects match. Keep all raw
failures and invalid suggestions in the report.

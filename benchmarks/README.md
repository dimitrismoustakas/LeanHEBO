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

The typed raw-result contract in `harness/` records wall and process CPU samples and rejects
comparisons unless every declared work field matches. The work contract includes the GP optimizer
and learning rate, parameter/optimizer reuse, `set_train_data` use, posterior batching, normalized
offspring generations, and candidate evaluations. Result writers refuse to overwrite an existing
raw record.

Prepare the audited upstream source and its dependencies in an isolated, ignored environment:

```bash
uv run python benchmarks/environments/prepare_upstream.py
```

The helper installs the exact development-only dependency set in
`environments/upstream-hebo.lock.txt`, verifies its SHA-256 digest, installs HEBO from the audited
checkout without re-resolving dependencies, and runs `uv pip check`.

Use `--checkout-only` to validate the exact commit and pinned setup-file digests without resolving
the old upstream dependency set. Upstream HEBO and Pymoo are development/reference inputs only;
neither is imported by `src/leanhebo`.

Run the structural search benchmark and focused harness tests with:

```bash
uv run pytest benchmarks/micro/benchmark_matched_search.py
uv run pytest benchmarks/tests
```

The toy suite defaults to one seed and intentionally small work. It proves that the runner works;
it is not evidence for a speed or quality claim. Suggestion timing is recorded separately as
initial Sobol, first model, and steady model phases:

```bash
uv run python -m benchmarks.quality.run_toy_suite --seeds 0
uv run python -m benchmarks.profiling.run_cprofile --seed 0
```

## Honest comparison lanes

Upstream Pymoo's native `n_gen=100` includes its initial population. The normalized work contract
therefore records 99 offspring generations and 10,000 candidate evaluations. LeanHEBO uses 99
offspring generations for the matched lane. Both implementations use pSGLD, 100 GP steps, one CPU
Torch thread, no posterior chunking, and a cold model lifecycle:

```powershell
uv run python -m benchmarks.quality.run_toy_suite `
  --implementation leanhebo --cases sphere-2d,mixed-3d --seeds 0,1,2,3,4 `
  --evaluation-budget 12 --batch-size 2 --random-samples 4 `
  --population-size 100 --generations 99 --gp-initial-steps 100 --gp-update-steps 100 `
  --gp-optimizer psgld --model-lifecycle cold --posterior-batch-size none `
  --torch-threads 1 --output-directory benchmarks/results/parity/leanhebo

benchmarks/.upstream/hebo-venv/Scripts/python.exe `
  -m benchmarks.quality.run_toy_suite `
  --implementation upstream-hebo --cases sphere-2d,mixed-3d --seeds 0,1,2,3,4 `
  --evaluation-budget 12 --batch-size 2 --random-samples 4 `
  --population-size 100 --generations 99 --gp-initial-steps 100 --gp-update-steps 100 `
  --gp-optimizer psgld --model-lifecycle cold --posterior-batch-size none `
  --torch-threads 1 --output-directory benchmarks/results/parity/upstream

uv run python -m benchmarks.quality.compare_results `
  --candidate benchmarks/results/parity/leanhebo `
  --baseline benchmarks/results/parity/upstream
```

The comparator pairs every case and seed, calls `assert_matched_work`, and refuses a mismatch. It
reports failures, duplicates, normalized regret, phase-specific ratios, and deterministic paired
bootstrap 95% intervals. Missing seeds and malformed or structurally stale work contracts also
fail closed.

The persistent lane intentionally measures LeanHEBO's shorter updates and reuse against upstream's
cold reconstruction. Generate Lean results with `--model-lifecycle persistent`, the desired update
steps and posterior batch size, then pass `--allow-changed-work` to the comparator. Its output is
prominently labeled `CHANGED-WORK`; its timing ratios are observations, not equivalent-work
speedups. Keep every raw failure and invalid or duplicate suggestion in the report.

## Fixed-history latency matrix

The fixed-history lane separates suggestion latency from an optimization trajectory. Each repeat
constructs a fresh optimizer, preloads an identical deterministic history outside the timed region,
and times exactly one cold model-based suggestion. The raw result stores the history SHA-256 and
validates the returned schema, bounds, batch size, uniqueness, and actual acquisition-search work.
Repeats are timing samples within one paired seed; they never reuse a fitted model.

The audited matrix covers 16/64/128 observations, continuous and mixed spaces, 5/20 public
parameters, and suggestion batches 1/4. Mixed spaces repeat a float/integer/categorical parameter
rotation, so dimension always means the public parameter count. The CLI intentionally defaults to
only the smallest cell. Run selected cells first rather than launching all 24 combinations
accidentally:

```powershell
uv run python -m benchmarks.latency.run_fixed_history `
  --implementation leanhebo --families continuous,mixed --dimensions 5 `
  --observations 16,64 --batch-sizes 1,4 --seeds 0,1,2 --repeats 3 `
  --output-directory benchmarks/results/fixed-history/leanhebo

benchmarks/.upstream/hebo-venv/Scripts/python.exe `
  -m benchmarks.latency.run_fixed_history `
  --implementation upstream-hebo --families continuous,mixed --dimensions 5 `
  --observations 16,64 --batch-sizes 1,4 --seeds 0,1,2 --repeats 3 `
  --output-directory benchmarks/results/fixed-history/upstream

uv run python -m benchmarks.latency.compare_fixed_history `
  --candidate benchmarks/results/fixed-history/leanhebo `
  --baseline benchmarks/results/fixed-history/upstream
```

Both commands default to the matched cold-work contract: CPU float32, one Torch thread, pSGLD with
100 fitting steps, population 100, 99 offspring generations, 10,000 acquisition candidates, and
unbatched posterior evaluation. The fixed-history comparator pairs case and seed, verifies the
history digest and matrix metadata, fails closed if any declared work differs, and reports this lane
as `MATCHED-WORK`. In these results,
`driver.suggest.first_model` is the fixed-history latency sample; regret is deliberately absent.
Repeats reconstruct model state but share process-level library caches. Use separate invocations or
seeds when process-cold startup behavior matters, and retain every raw timing sample.

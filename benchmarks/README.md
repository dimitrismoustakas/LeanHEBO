# Benchmark

No benchmark result has been published yet.

The study asks one question: at the same CARP-S evaluation budgets, does LeanHEBO match
upstream HEBO's optimization quality while spending less time in `ask` and `tell`?

It uses the 13 flat tasks in [`tasks.json`](carps/tasks.json): eight BBOB tasks and five
YAHPO surrogate HPO tasks from the CARP-S 1.1.0 black-box test set. Conditional tasks and the
HPOBench task are excluded because LeanHEBO does not support conditional spaces and HPOBench
requires a separate installation path. Each optimizer uses its defaults, suggests one point at
a time on one CPU thread, and runs 20 seeds with CARP-S's task budgets: 34,900 evaluations per
optimizer.

Quality is CARP-S's pooled per-task min-max normalized incumbent cost; lower is better. The
figure averages the fixed task set within each seed and shows a 95% interval over the 20 seeds.
The table reports the paired quality difference, cumulative `ask` plus `tell` time, and the
HEBO/LeanHEBO time ratio. Failed or incomplete runs stop the analysis instead of being dropped.

## Run it

The benchmark has a separate locked environment. It adds nothing to LeanHEBO's dependencies.
The first setup builds CARP-S from source and can take about 15 minutes on Windows.

```powershell
python benchmarks/carps/prepare.py
$python = (Resolve-Path 'benchmarks/.carps/venv/Scripts/python.exe').Path
$env:CARPS_TASK_DATA_DIR = (Resolve-Path 'benchmarks/.carps/task_data').Path
$env:OMP_NUM_THREADS = $env:MKL_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = '1'
```

Check the two real adapters with one trial each:

```powershell
$common = @(
  'hydra.searchpath=[pkg://leanhebo_carps.configs]',
  '+task/subselection/blackbox/test=subset_bbob_2_12_2',
  'seed=1',
  'task.optimization_resources.n_trials=1',
  'baserundir=benchmarks/.carps/smoke'
)
uv run --no-project --python $python python -X utf8 -m carps.run `
  '+optimizer/leanhebo=config' @common
uv run --no-project --python $python python -X utf8 -m carps.run `
  '+optimizer/hebo_timed=config' @common
```

Run the full comparison only when its cost is intentional:

```powershell
$tasks = ((Get-Content 'benchmarks/carps/tasks.json' -Raw | ConvertFrom-Json).config) -join ','
$seeds = (1..20) -join ','
$common = @(
  'hydra.searchpath=[pkg://leanhebo_carps.configs]',
  "+task/subselection/blackbox/test=$tasks",
  "seed=$seeds",
  'baserundir=benchmarks/.carps/runs'
)
uv run --no-project --python $python python -X utf8 -m carps.run --multirun `
  '+optimizer/leanhebo=config' @common
uv run --no-project --python $python python -X utf8 -m carps.run --multirun `
  '+optimizer/hebo_timed=config' @common

uv run --no-project --python $python python -X utf8 -m carps.analysis.gather_data `
  'benchmarks/.carps/runs' --n_processes=1 --outdir='benchmarks/.carps/gathered'
uv run --no-project --python $python python -X utf8 'benchmarks/carps/analyze.py' `
  --logs 'benchmarks/.carps/gathered/logs.parquet' `
  --runs 'benchmarks/.carps/runs' --output 'benchmarks/.carps/carps.png'
```

For a later hyperparameter study, add a new optimizer config with a unique `optimizer_id` and
run only that config. Tune on CARP-S development tasks, then evaluate the chosen setting on this
test set. Existing test results remain reusable while the task list and benchmark lock stay
unchanged; gather and analyze again to include the new result.

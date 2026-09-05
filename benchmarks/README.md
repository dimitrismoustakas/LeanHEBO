# Benchmarks

[HEBO comparison](results/hebo-comparison/report.md): LeanHEBO 0.3.0 versus HEBO 0.3.6,
13 flat CARP-S tasks, 20 seeds. All 520 attempts are included; HEBO's three failed
LCBench runs retain their best evaluated point. This is a historical result.

## Run and compare

The separate environment contains CARP-S, upstream HEBO, and YAHPO data. Setup requires
`uv` and Git; the first build can take about 15 minutes on Windows.

```powershell
python benchmarks/carps/prepare.py
$python = 'benchmarks/.carps/venv/Scripts/python.exe'
& $python benchmarks/carps/run.py --optimizer leanhebo --seeds (1..20) --output benchmarks/.carps/runs
& $python benchmarks/carps/run.py --optimizer hebo --seeds (1..20) --output benchmarks/.carps/runs
& $python benchmarks/carps/analyze.py --runs benchmarks/.carps/runs --reference HEBO --output benchmarks/results/comparison
```

On Linux, use `benchmarks/.carps/venv/bin/python`. Runs use one CPU thread and one suggestion
at a time. `--trials 2` gives a short smoke check. Each run writes one `run.jsonl` containing
settings, evaluated configurations, costs, and ask/tell times. Existing runs are never overwritten.

Select another LeanHEBO checkout with `--source ../LeanHEBO-conditional --label LeanHEBO-conditional`.
`--optimizer random` runs CARP-S random search. For another model or settings, pass a YAML file
with `name` and `optimizer` entries, following the small [optimizer configs](carps/leanhebo_carps/configs/optimizer).
Any sequential, single-objective CARP-S adapter can use the same runner and analysis.

## Tasks

Pass a list with `--tasks`; each entry needs only `name` and `n_trials`.

- [Development](carps/development_tasks.json), the default: 13 tasks, including four 8D BBOB
  problems and conditional YAHPO spaces. The retained one-seed pilot supports further screening;
  LCBench `168335` still needs a HEBO comparison before deciding whether to replace it.
- [Flat comparison](carps/tasks.json): the original 13-task CARP-S test selection.
- [Conditional holdout](carps/conditional_tasks.json): five SVM/XGBoost tasks.
- [Synthetic diagnostics](carps/synthetic_tasks.json): activation, shared roots, eight exclusive
  branches, a narrow active region (`gate > 0.9`), and mixed log/integer parameters. Each objective
  is a sum of nonnegative offsets and weighted squares, with an attainable minimum of zero.

YAHPO uses negative accuracy at maximum fidelity. A task's optional `metric` selects another
YAHPO objective, such as `val_cross_entropy` for LCBench. Choose tasks and metrics on development
data before the final comparison. Task construction uses each benchmark's own ConfigSpace.

## Metrics

Aggregate quality is the probability of beating the reference optimizer: compare every pair of
runs within each task, count ties as half, then average tasks equally. 50% is neutral. Anytime
probability averages 20 budget fractions from 5% to 100%. This uses the
[probability of improvement](https://github.com/google-research/rliable#probability-of-improvement)
metric; it measures how often an optimizer wins without normalizing task costs.
Per-task plots show median and interquartile range of native cost, or regret against the known
optimum for BBOB and synthetic tasks. The table also gives median paired cost differences.

Failed runs carry their best evaluated cost through the remaining budget and count as failures.
A run with no evaluated point has infinite cost. Timing includes only task/seed pairs completed
by every compared optimizer; speedup is the median of paired time ratios. These are descriptive
comparisons, with no confidence interval or significance claim. Analysis requires the same tasks,
seeds, budgets, and objectives across optimizers.

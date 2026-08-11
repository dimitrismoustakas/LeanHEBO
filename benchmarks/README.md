# Benchmarks

LeanHEBO does not have a published performance claim yet.

## Planned Bayesmark benchmark

The benchmark will answer one question:

> With the same Bayesmark evaluation budget, does LeanHEBO match upstream HEBO's
> optimization quality while using less optimizer time?

The comparison will use LeanHEBO and the pinned upstream HEBO implementation on the
108 tasks from the HEBO paper: 20 seeds, 16 batches of 8 evaluations, and one CPU
thread. Each optimizer will use its recommended configuration. The evaluation budget
is equal; their internal implementations do not need to perform identical work.

The result will be one figure with two plots:

1. Bayesmark normalized mean loss versus evaluations, with 95% intervals. Lower is better.
2. Cumulative `suggest` plus `observe` time versus evaluations.

A small table will report final quality, optimizer time, the difference between the two
implementations, and failed runs. Before running it, we will define how close the final scores
must be to count as equal quality. Until this study is complete, the repository will not claim
equal quality or a general speedup.

## Existing timing check

`benchmarks.latency.run_fixed_history` measures one cold model-based suggestion from
the same synthetic observation history in both implementations. It is useful for
finding suggestion overhead, but it does not measure optimization quality or a full
optimization run and is not a release benchmark.

The toy objectives under `benchmarks.quality` only test the benchmark code. They are
not evidence about optimizer quality.

The upstream comparison is pinned to HEBO commit
`ee6112d39d1a9e9703fecaf9057193e1ec9dae72`.

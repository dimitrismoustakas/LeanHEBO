# Contributing

Thank you for helping improve LeanHEBO. Keep changes focused, add tests for observable behavior,
and explain compatibility or performance tradeoffs in the pull request.

## Development setup

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and run:

```console
uv python install 3.11
uv sync --locked --all-groups --all-extras --python 3.11
```

Before submitting a change, run the same checks as CI:

```console
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src/leanhebo benchmarks
uv run --locked pytest --cov=leanhebo --cov-report=xml
uv run --locked pytest benchmarks/tests benchmarks/micro/benchmark_matched_search.py
uv build --no-sources
```

## Scope and provenance

LeanHEBO intentionally targets the single-objective exact-GP HEBO path described in the README.
Discuss substantial scope expansions before implementing them. Preserve Huawei copyright headers
where implementation text is substantially derived from upstream HEBO. For other borrowed or
adapted work, document the source and license in the pull request and update `ORIGIN.md` or
`NOTICE.md` when appropriate. Do not contribute code with unclear or incompatible provenance.

## Performance work

Performance claims must follow [`benchmarks/README.md`](benchmarks/README.md): use the pinned
upstream baseline, matched algorithmic work, multiple seeds, and all samples including failures.
Include candidate and baseline commits, exact commands and configuration, hardware, OS, Python,
Torch and GPyTorch versions, device, dtype, thread counts, uncertainty intervals, and a location
for the unmodified raw data. Label changed-work comparisons as such; do not present them as
equivalent-work speedups.

Reusable benchmark code and tests are welcome. Do not commit raw benchmark results, generated
reports, profiles, coverage output, or other run artifacts. Attach reproducibility data to the
GitHub issue or place it in an external durable location instead.

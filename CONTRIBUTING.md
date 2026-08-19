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
uv run --locked mypy src/leanhebo
uv run --locked pytest
uv run --locked pytest benchmarks/tests
uv build --no-sources
```

## Scope and provenance

LeanHEBO intentionally targets the single-objective exact-GP HEBO path described in the README.
Discuss substantial scope expansions before implementing them. Preserve Huawei copyright headers
where implementation text is substantially derived from upstream HEBO. For other borrowed or
adapted work, document the source and license in the pull request and update `NOTICE.md` when
appropriate. Do not contribute code with unclear or incompatible provenance.

## Performance work

For a performance change, report the question being measured, before and after results, exact
commands, commits, configuration, and hardware. Use enough repeated runs to show that the result
is stable, include failures, and link the raw data. If the two runs do different work, say so.

Commit reusable benchmark code and tests, not run artifacts such as raw results, reports,
profiles, or coverage output. Attach those to the GitHub issue or link them.

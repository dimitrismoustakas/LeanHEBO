# LeanHEBO

LeanHEBO is a tensor-native, low-latency implementation of the established HEBO Bayesian
optimization strategy. It focuses on the single-objective exact-GP path, persistent model
state, mixed-variable search, and explicit CPU or CUDA execution.

LeanHEBO is an independent project. It is not affiliated with or endorsed by Huawei and
does not claim a new optimization algorithm. See [NOTICE.md](NOTICE.md) and
[ORIGIN.md](ORIGIN.md) for provenance.

The package is under active initial development. Its first release targets continuous,
log-continuous, integer, stepped-integer, power/exponent-integer, Boolean, and categorical
variables; batch and contextual suggestions; HEBO output warping; MACE acquisition; and a
tensor-native NSGA-II search.

## Quick start

```python
import numpy as np

from leanhebo import LeanHEBO
from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
from leanhebo.space import Bool, Categorical, Float, Integer, Space

space = Space(
    Float("learning_rate", 1e-5, 1e-1, log=True),
    Integer("depth", 1, 20),
    Categorical("activation", ("relu", "gelu", "silu")),
    Bool("use_bias"),
)
config = LeanHEBOConfig(
    runtime=RuntimeConfig(device="cpu", dtype="float32", seed=7),
    gp=GPConfig(update_steps=12, full_refit_interval=20),
    search=SearchConfig(population_size=128, generations=80),
)
optimizer = LeanHEBO(space, config=config)

for _ in range(30):
    candidates = optimizer.suggest(4)
    values = np.asarray([objective(row) for row in candidates.to_records()])
    optimizer.observe(candidates, values)
```

Passing the original `CandidateBatch` to `observe` reuses its encoded tensors. Records,
NumPy arrays, Pandas DataFrames, and Polars DataFrames are also accepted at the boundary and
encoded once.

Contextual values are fixed without a DataFrame round trip:

```python
candidates = optimizer.suggest(8, fix_input={"use_bias": True})
```

Pandas and Polars are optional runtime adapters:

```console
uv sync --extra pandas
uv sync --extra polars
```

```python
frame = candidates.to_polars()  # or candidates.to_pandas()
optimizer.observe(candidates, objective(frame))
```

## Runtime behavior

Runtime code keeps encoded observations, transforms, GP state, acquisition evaluation, random
streams, and evolutionary populations in Torch. NumPy and optional table libraries are boundary
formats only. The GP and likelihood survive ordinary updates; new training data are installed
with GPyTorch's persistent exact-GP API and receive the configured short update schedule.

CPU is the default. CUDA is selected explicitly with `RuntimeConfig(device="cuda")`; LeanHEBO
does not silently move work between devices. Both `float32` and `float64` are supported.

Importing LeanHEBO does not change Torch or BLAS thread counts. Applications that want an
explicit process-wide policy should call the startup helper before other Torch work:

```python
from leanhebo.runtime import configure_process

configure_process(torch_num_threads=1, torch_num_interop_threads=1)
```

Every configuration group has one documented set of defaults and serializes with
`optimizer.config.to_dict()`. There are no hidden execution modes or preset profiles.

## Checkpointing

```python
optimizer.save("run.leanhebo")
restored = LeanHEBO.load("run.leanhebo", map_location="cpu")
```

The versioned checkpoint contains only LeanHEBO-defined state, tensors, and primitive schema
values. It restores observations, transforms, model and likelihood parameters, optimizer state,
Sobol progress, and independent Torch generator states. User functions are never pickled.

## Scope

The initial implementation is intentionally narrow: one user objective, an exact Gaussian
process, MACE, and mixed-variable NSGA-II. Random forests, CatBoost, neural ensembles, sparse
GPs, constrained user objectives, and general multi-objective user objectives are not part of
the first release.

No performance or quality claim should be made without the pinned, matched-work benchmark and
multi-seed quality suite described in [benchmarks/README.md](benchmarks/README.md).

## Development

```console
uv sync --all-groups --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/leanhebo
uv build
```

The repository uses Python 3.11+, a `src/` layout, the pure-Python `uv_build` backend, and a
committed CPU development lock. Optional accelerator environments should override only the
explicit Torch package index.

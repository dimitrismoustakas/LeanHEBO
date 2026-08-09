# LeanHEBO

LeanHEBO is a tensor-native, low-latency implementation of the established HEBO Bayesian
optimization strategy. It focuses on the single-objective exact-GP path, persistent model
state, mixed-variable search, and explicit CPU or CUDA execution.

LeanHEBO is an independent project. It is not affiliated with or endorsed by Huawei and
does not claim a new optimization algorithm. See
[NOTICE.md](https://github.com/dimitrismoustakas/LeanHEBO/blob/main/NOTICE.md) and
[ORIGIN.md](https://github.com/dimitrismoustakas/LeanHEBO/blob/main/ORIGIN.md) for provenance.

The package is under active initial development. Its first release targets continuous,
log-continuous, integer, stepped-integer, power/exponent-integer, Boolean, and categorical
variables; batch and contextual suggestions; HEBO output warping; MACE acquisition; and a
tensor-native NSGA-II search.

## Installation

LeanHEBO requires Python 3.11 or newer; the initial release is tested on Python 3.11 through
3.13. Install the released package with either pip or uv:

```console
python -m pip install leanhebo
```

```console
uv add leanhebo
```

Pandas and Polars adapters are optional extras:

```console
python -m pip install "leanhebo[pandas,polars]"
# or: uv add "leanhebo[pandas,polars]"
```

CPU is the primary supported path for the initial alpha release. CUDA execution is available,
but accelerator support depends on the Torch build and the local CUDA stack. Install the
appropriate Torch build using the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
before installing LeanHEBO; the package does not silently switch between CPU and CUDA.

## Quick start

LeanHEBO minimizes the objective values supplied to `observe`. This complete example optimizes
a small mixed-variable loss:

```python
import numpy as np

from leanhebo import LeanHEBO
from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def objective(row: dict[str, object]) -> float:
    """Return a loss: LeanHEBO minimizes the values passed to observe."""
    x = float(row["x"])
    depth = int(row["depth"])
    activation_penalty = 0.0 if row["activation"] == "gelu" else 0.25
    bias_penalty = 0.0 if bool(row["use_bias"]) else 0.1
    return (x - 0.3) ** 2 + (depth - 6) ** 2 / 25 + activation_penalty + bias_penalty


space = Space(
    Float("x", -2.0, 2.0),
    Integer("depth", 1, 12),
    Categorical("activation", ("relu", "gelu", "silu")),
    Bool("use_bias"),
)
config = LeanHEBOConfig(
    runtime=RuntimeConfig(seed=7),
    # Small settings keep this example quick; use the defaults for real optimization.
    gp=GPConfig(initial_steps=15, update_steps=3),
    search=SearchConfig(population_size=32, generations=15),
)
optimizer = LeanHEBO(space, config=config)

for _ in range(3):
    candidates = optimizer.suggest(3)
    losses = np.asarray([objective(row) for row in candidates.to_records()])
    optimizer.observe(candidates, losses)

print(optimizer.best_x.to_records()[0], optimizer.best_y)
```

Passing the original `CandidateBatch` to `observe` avoids re-encoding; the observation store takes
one owned tensor snapshot so later caller mutation cannot rewrite history. Records, NumPy arrays,
Pandas DataFrames, and Polars DataFrames are also accepted at the boundary and encoded once.

Contextual values are fixed without a DataFrame round trip:

```python
candidates = optimizer.suggest(8, fix_input={"use_bias": True})
```

```python
candidates = optimizer.suggest(3)
frame = candidates.to_polars()  # or candidates.to_pandas()
losses = np.asarray([objective(row) for row in candidates.to_records()])
optimizer.observe(frame, losses)
```

## Runtime behavior

Runtime code keeps encoded observations, transforms, GP state, acquisition evaluation, random
streams, and evolutionary populations in Torch. NumPy and optional table libraries are boundary
formats only. The GP and likelihood survive ordinary updates; new training data are installed
with GPyTorch's persistent exact-GP API and receive the configured short update schedule.

CPU is the default. CUDA is selected explicitly with `RuntimeConfig(device="cuda")`; LeanHEBO
does not silently move work between devices. Both `float32` and `float64` are supported. The
project is currently alpha software: CPU behavior receives the broadest release validation,
while CUDA compatibility can vary across Torch, driver, and device combinations.

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
multi-seed quality suite described in
[benchmarks/README.md](https://github.com/dimitrismoustakas/LeanHEBO/blob/main/benchmarks/README.md).

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

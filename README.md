# LeanHEBO

LeanHEBO is a PyTorch implementation of HEBO. It uses an exact Gaussian process to minimize a
single unconstrained objective. Search spaces can contain bounded floats (linear or logarithmic),
integers (linear, stepped, logarithmic, or powers of a base), Boolean values, and categories.
Parameters can depend on other choices, and you can request a batch of suggestions with some
inputs fixed.

Upstream attribution is in [NOTICE.md](https://github.com/dimitrismoustakas/LeanHEBO/blob/main/NOTICE.md).

## Install

LeanHEBO requires Python 3.11 or newer and is tested on Python 3.11–3.13.

```console
python -m pip install leanhebo
# or
uv add leanhebo
```

CPU is the default device and the one used for release testing. Set `RuntimeConfig(device="cuda")`
to use experimental CUDA support.

## Quick start

LeanHEBO minimizes the values passed to `observe`.

```python
import numpy as np

from leanhebo import LeanHEBO
from leanhebo.config import GPConfig, LeanHEBOConfig, RuntimeConfig, SearchConfig
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def loss(row: dict[str, object]) -> float:
    return (
        (float(row["x"]) - 0.3) ** 2
        + (int(row["depth"]) - 6) ** 2 / 25
        + (0.0 if row["activation"] == "gelu" else 0.25)
        + (0.0 if bool(row["use_bias"]) else 0.1)
    )


space = Space(
    Float("x", -2.0, 2.0),
    Integer("depth", 1, 12),
    Categorical("activation", ("relu", "gelu", "silu")),
    Bool("use_bias"),
)
config = LeanHEBOConfig(
    runtime=RuntimeConfig(seed=7),
    gp=GPConfig(initial_steps=15, update_steps=3),
    search=SearchConfig(population_size=32, generations=15),
)
optimizer = LeanHEBO(space, config=config)

for _ in range(3):
    candidates = optimizer.suggest(3)
    optimizer.observe(candidates, np.asarray([loss(row) for row in candidates.to_records()]))

print(optimizer.best_x.to_records()[0], optimizer.best_y)
```

The reduced fitting and search settings keep this example quick; the defaults do more work.

Pass the returned `CandidateBatch` directly to `observe` to avoid re-encoding. Records, column
mappings, and NumPy arrays are also accepted at the boundary. Fix contextual values when suggesting
with:

```python
candidates = optimizer.suggest(8, fix_input={"use_bias": True})
```

Non-finite outcomes are dropped by default. Set `nonfinite_policy="raise"` in
`LeanHEBOConfig` to reject them instead.

After the initial design, the default search scores every legal configuration when the
current space or fixed-input context has at most 8,192 configurations. It uses the usual
MACE Pareto front and batch selection. Larger and continuous domains use NSGA-II.
Custom evolutionary settings select NSGA-II. Set `SearchConfig(exhaustive_limit=n)` to
choose an explicit cutoff alongside those settings, or use `exhaustive_limit=0` for
NSGA-II at every size.

## Conditional spaces

Attach an `active_when` condition to each child parameter:

```python
from leanhebo import LeanHEBO
from leanhebo.space import Categorical, Eq, Float, In, Integer, Space

space = Space(
    Categorical("booster", ("gblinear", "gbtree", "dart")),
    Float("reg_lambda", 1e-4, 1.0, log=True),
    Integer("max_depth", 2, 12, active_when=In("booster", ("gbtree", "dart"))),
    Float("rate_drop", 0.0, 1.0, active_when=Eq("booster", "dart")),
)
optimizer = LeanHEBO(space)
candidates = optimizer.suggest(8, fix_input={"rate_drop": 0.2})
records = candidates.to_records()
```

`to_records()` returns sparse rows and omits inactive parameters. The encoded candidate tensors
remain rectangular; `candidates.activity` is their Boolean parameter-activity matrix. Fixing
`rate_drop` supplies its value whenever it is active, but does not force `booster="dart"`; other
branches still omit it.

## Runtime and checkpoints

Numerical observations, transformations, GP state, acquisition evaluation, random streams, and
evolutionary populations use Torch. The exact GP persists across ordinary updates. Both `float32`
and `float64` are supported.

LeanHEBO uses the process’s Torch and BLAS thread settings.

Save and resume without pickling user functions:

```python
optimizer.save("run.leanhebo")
restored = LeanHEBO.load("run.leanhebo", map_location="cpu")
```

The checkpoint restores observations, transformations, model and optimizer state, Sobol progress,
and independent Torch generator states.

## Benchmarks

The comparison of LeanHEBO 0.3.0 and HEBO 0.3.6 covers 13 CARP-S tasks with 20 seeds per task.
For results and commands, see the
[benchmark guide](https://github.com/dimitrismoustakas/LeanHEBO/blob/main/benchmarks/README.md).

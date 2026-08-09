# SPDX-License-Identifier: MIT

"""Explicit work budgets used to prevent misleading timing comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeAlias

Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class WorkBudget:
    """Algorithmic work declared by one benchmark result.

    ``None`` means that the implementation did not expose or declare that dimension. It is not a
    wildcard: an unknown value does not match a known value. This makes accidental comparisons of
    a warm-update run with an upstream cold-refit run fail closed.
    """

    objective_evaluations: int
    batch_size: int
    population_size: int | None = None
    generations: int | None = None
    gp_initial_steps: int | None = None
    gp_update_steps: int | None = None
    full_refit_interval: int | None = None
    posterior_batch_size: int | None = None
    random_samples: int | None = None

    def __post_init__(self) -> None:
        non_negative = (
            "objective_evaluations",
            "generations",
            "gp_initial_steps",
            "gp_update_steps",
        )
        positive = (
            "batch_size",
            "population_size",
            "full_refit_interval",
            "posterior_batch_size",
            "random_samples",
        )
        for field_name in non_negative:
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        for field_name in positive:
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or value < 1):
                raise ValueError(f"{field_name} must be a positive integer or None")

    @property
    def expected_search_evaluations(self) -> int | None:
        """Candidate rows evaluated by a full-population fixed-generation search."""

        if self.population_size is None or self.generations is None:
            return None
        return self.population_size * (self.generations + 1)

    def to_dict(self) -> dict[str, Scalar]:
        """Return the stable JSON representation used by result files."""

        return asdict(self)


def assert_matched_work(left: WorkBudget, right: WorkBudget) -> None:
    """Raise with a field-by-field explanation unless two budgets match exactly."""

    differences = {
        name: (left_value, right_value)
        for name, left_value in left.to_dict().items()
        if left_value != (right_value := right.to_dict()[name])
    }
    if differences:
        details = ", ".join(
            f"{name}: {left_value!r} != {right_value!r}"
            for name, (left_value, right_value) in differences.items()
        )
        raise ValueError(f"benchmark work is not matched ({details})")

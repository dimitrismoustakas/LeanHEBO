# SPDX-License-Identifier: MIT

"""Explicit work budgets used to prevent misleading timing comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TypeAlias

Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class WorkBudget:
    """Algorithmic work declared by one benchmark result.

    ``None`` means that the implementation did not expose or declare that dimension. It is not a
    wildcard: an unknown value does not match a known value. This makes accidental comparisons of
    a warm-update run with an upstream cold-refit run fail closed.

    ``generations`` is normalized to offspring generations: the initial population is excluded.
    ``search_candidate_evaluations`` includes the initial population and every offspring batch.
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
    search_candidate_evaluations: int | None = None
    gp_optimizer: str | None = None
    learning_rate: float | None = None
    reuse_parameters: bool | None = None
    reuse_optimizer_state: bool | None = None
    use_set_train_data: bool | None = None
    device: str | None = None
    dtype: str | None = None
    torch_threads: int | None = None

    def __post_init__(self) -> None:
        required = ("objective_evaluations", "batch_size")
        optional = (
            "population_size",
            "generations",
            "gp_initial_steps",
            "gp_update_steps",
            "full_refit_interval",
            "posterior_batch_size",
            "random_samples",
            "search_candidate_evaluations",
            "torch_threads",
        )
        for field_name in required:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        for field_name in optional:
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{field_name} must be an integer or None")

        non_negative = (
            "objective_evaluations",
            "generations",
            "gp_initial_steps",
            "gp_update_steps",
            "search_candidate_evaluations",
        )
        positive = (
            "batch_size",
            "population_size",
            "full_refit_interval",
            "posterior_batch_size",
            "random_samples",
            "torch_threads",
        )
        for field_name in non_negative:
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        for field_name in positive:
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be a positive integer or None")
        if self.gp_optimizer is not None and (
            not isinstance(self.gp_optimizer, str) or not self.gp_optimizer
        ):
            raise TypeError("gp_optimizer must be a non-empty string or None")
        if self.learning_rate is not None and (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite or None")
        for field_name in ("reuse_parameters", "reuse_optimizer_state", "use_set_train_data"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean or None")
        for field_name in ("device", "dtype"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{field_name} must be a non-empty string or None")

        expected = self.expected_search_evaluations
        if (
            expected is not None
            and self.search_candidate_evaluations is not None
            and self.search_candidate_evaluations != expected
        ):
            raise ValueError(
                "search_candidate_evaluations must equal population_size * (generations + 1)"
            )

    @property
    def expected_search_evaluations(self) -> int | None:
        """Candidate rows evaluated by a full-population fixed-generation search."""

        if self.population_size is None or self.generations is None:
            return None
        return self.population_size * (self.generations + 1)

    def to_dict(self) -> dict[str, Scalar]:
        """Return the stable JSON representation used by result files."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> WorkBudget:
        """Load the exact work contract stored in a raw benchmark result."""

        return cls(**dict(value))  # type: ignore[arg-type]


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

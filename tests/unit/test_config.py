# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from leanhebo.config import (
    AcquisitionConfig,
    GPConfig,
    LeanHEBOConfig,
    RuntimeConfig,
    SearchConfig,
    WarpConfig,
)


def test_configuration_round_trip() -> None:
    config = LeanHEBOConfig(
        random_samples=5,
        runtime=RuntimeConfig(dtype="float64", seed=17),
        gp=GPConfig(optimizer="adam", initial_steps=3),
        search=SearchConfig(population_size=16, generations=4),
    )
    assert LeanHEBOConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RuntimeConfig(acquisition_batch_size=0), "acquisition_batch_size"),
        (lambda: GPConfig(full_refit_growth_factor=1.0), "growth_factor"),
        (lambda: GPConfig(use_fantasy_updates=True), "not implemented"),
        (lambda: GPConfig(learning_rate=float("nan")), "learning_rate"),
        (
            lambda: GPConfig(noise_lower_bound=0.01, noise_initial=0.01),
            "noise_initial.*strictly greater",
        ),
        (lambda: RuntimeConfig(dtype="float16"), "dtype"),  # type: ignore[arg-type]
        (lambda: WarpConfig(lambda_lower_bound=float("nan")), "lambda bounds"),
        (lambda: AcquisitionConfig(kappa=float("nan")), "kappa"),
        (lambda: SearchConfig(population_size=1), "population_size"),
        (lambda: LeanHEBOConfig(random_samples=1), "random_samples"),
    ],
)
def test_invalid_configuration_is_rejected(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]

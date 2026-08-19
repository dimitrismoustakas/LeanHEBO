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
        gp=GPConfig(initial_steps=3),
        search=SearchConfig(population_size=16, generations=4),
    )
    assert LeanHEBOConfig.from_dict(config.to_dict()) == config


def test_unknown_top_level_configuration_field_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"unknown LeanHEBOConfig field.*'serach'"):
        LeanHEBOConfig.from_dict({"serach": {}})


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RuntimeConfig(acquisition_batch_size=0), "acquisition_batch_size"),
        (lambda: GPConfig(full_refit_growth_factor=1.0), "growth_factor"),
        (
            lambda: GPConfig(use_fantasy_updates=True),
            "fantasy updates require update_steps=0",
        ),
        (
            lambda: GPConfig(
                update_steps=0,
                use_fantasy_updates=True,
                reuse_parameters=False,
            ),
            "reuse_parameters=True",
        ),
        (lambda: GPConfig(learning_rate=float("nan")), "learning_rate"),
        (lambda: GPConfig(jitter=0.0), "jitter"),
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


def test_fantasy_updates_allow_scheduled_output_refits() -> None:
    gp = GPConfig(update_steps=0, use_fantasy_updates=True)

    config = LeanHEBOConfig(gp=gp, warp=WarpConfig(refit_interval=3))
    assert config.gp.use_fantasy_updates

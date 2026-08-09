# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.gp import ExactGPSurrogate
from leanhebo.gp.kernel import (
    MixedFeatureExtractor,
    build_kernel,
    initialize_numeric_lengthscales,
)
from leanhebo.gp.optimizer import PreconditionedSGLD
from leanhebo.runtime.rng import make_generator


def _surrogate(*, categories: tuple[int, ...] = ()) -> ExactGPSurrogate:
    return ExactGPSurrogate(
        num_continuous=1,
        category_sizes=categories,
        config=GPConfig(
            optimizer="adam",
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
        ),
        runtime=RuntimeConfig(seed=4),
        generator=make_generator("cpu", 4),
    )


def test_persistent_update_reuses_model_and_changes_train_data() -> None:
    gp = _surrogate()
    continuous = torch.linspace(0, 1, 6).reshape(-1, 1)
    categorical = torch.empty((6, 0), dtype=torch.long)
    targets = torch.sin(continuous[:, 0] * 3)
    initial = gp.fit(continuous, categorical, targets, transform_version=1)
    model_id = id(gp.model)

    updated_continuous = torch.cat((continuous, torch.tensor([[0.37]])))
    updated_categorical = torch.empty((7, 0), dtype=torch.long)
    updated_targets = torch.sin(updated_continuous[:, 0] * 3)
    update = gp.fit(
        updated_continuous,
        updated_categorical,
        updated_targets,
        transform_version=1,
    )

    assert initial.kind == "initial"
    assert update.kind == "update"
    assert id(gp.model) == model_id
    assert gp.update_count == 1
    assert gp.model is not None
    assert gp.model.train_targets.shape == (7,)


def test_input_scaler_fits_the_observed_range_instead_of_design_bounds() -> None:
    gp = _surrogate()
    continuous = torch.tensor([[40.0], [50.0], [60.0]])
    categorical = torch.empty((3, 0), dtype=torch.long)
    targets = torch.tensor([1.0, 0.0, 1.0])

    gp.fit(continuous, categorical, targets, transform_version=1)

    torch.testing.assert_close(
        gp.train_continuous,
        torch.tensor([[-1.0], [0.0], [1.0]]),
    )
    torch.testing.assert_close(gp.input_scaler.data_min_, torch.tensor([40.0]))  # type: ignore[union-attr]
    torch.testing.assert_close(gp.input_scaler.data_max_, torch.tensor([60.0]))  # type: ignore[union-attr]
    assert gp.input_scaler_version == 1


def test_observed_range_scaler_and_predictions_survive_state_round_trip() -> None:
    gp = _surrogate()
    continuous = torch.tensor([[40.0], [50.0], [60.0], [65.0]])
    categorical = torch.empty((4, 0), dtype=torch.long)
    targets = torch.tensor([1.0, 0.0, 1.0, 2.25])
    query = torch.tensor([[45.0], [70.0]])
    query_categories = torch.empty((2, 0), dtype=torch.long)
    gp.fit(continuous, categorical, targets, transform_version=3)
    expected = gp.predict(query, query_categories)

    restored = _surrogate()
    restored.load_state_dict(gp.state_dict())
    actual = restored.predict(query, query_categories)

    assert restored.input_scaler_version == gp.input_scaler_version
    torch.testing.assert_close(restored.input_scaler.data_min_, torch.tensor([40.0]))  # type: ignore[union-attr]
    torch.testing.assert_close(restored.input_scaler.data_max_, torch.tensor([65.0]))  # type: ignore[union-attr]
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_categorical_only_gp_uses_checkpointed_identity_input_scaler() -> None:
    gp = ExactGPSurrogate(
        num_continuous=0,
        category_sizes=(2,),
        config=GPConfig(optimizer="adam", initial_steps=1, update_steps=1),
        runtime=RuntimeConfig(seed=6),
        generator=make_generator("cpu", 6),
    )
    continuous = torch.empty((4, 0))
    categorical = torch.tensor([[0], [1], [0], [1]])
    targets = torch.tensor([0.0, 1.0, 0.1, 0.9])
    gp.fit(continuous, categorical, targets, transform_version=1)
    expected = gp.predict(continuous[:2], categorical[:2])

    restored = ExactGPSurrogate(
        num_continuous=0,
        category_sizes=(2,),
        config=gp.config,
        runtime=gp.runtime,
        generator=make_generator("cpu", 7),
    )
    restored.load_state_dict(gp.state_dict())
    actual = restored.predict(continuous[:2], categorical[:2])

    assert restored.input_scaler_version == 1
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_mixed_gp_posterior_and_state_round_trip() -> None:
    gp = _surrogate(categories=(3,))
    continuous = torch.linspace(0, 1, 9).reshape(-1, 1)
    categorical = (torch.arange(9) % 3).reshape(-1, 1)
    targets = continuous[:, 0] + categorical[:, 0].float() * 0.2
    gp.fit(continuous, categorical, targets, transform_version=2)
    mean, variance, noise = gp.predict(continuous[:3], categorical[:3])
    assert mean.shape == variance.shape == (3,)
    assert noise.ndim == 0
    assert torch.isfinite(mean).all()
    assert torch.all(variance > 0)

    restored = _surrogate(categories=(3,))
    restored.load_state_dict(gp.state_dict())
    restored_mean, restored_variance, _ = restored.predict(continuous[:3], categorical[:3])
    torch.testing.assert_close(restored_mean, mean)
    torch.testing.assert_close(restored_variance, variance)


def test_numeric_lengthscale_uses_median_of_all_pairwise_distances() -> None:
    extractor = MixedFeatureExtractor(1, ())
    kernel = build_kernel(num_continuous=1, feature_extractor=extractor, ard=True)
    values = torch.tensor([[0.0], [1.0], [2.0], [100.0]])

    initialize_numeric_lengthscales(
        kernel,
        values,
        sample_limit=4,
        lower_bound=0.02,
        generator=make_generator("cpu", 2),
    )

    # Pairwise distances are [1, 2, 100, 1, 99, 98], whose lower median is 2.
    torch.testing.assert_close(kernel.base_kernel.lengthscale.reshape(-1), torch.tensor([2.0]))


def test_scheduled_full_refit_cadence_resets_after_each_refit() -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
            optimizer="adam",
            initial_steps=0,
            update_steps=0,
            full_refit_interval=2,
            full_refit_growth_factor=None,
        ),
        runtime=RuntimeConfig(seed=4),
        generator=make_generator("cpu", 4),
    )
    categorical = torch.empty((5, 0), dtype=torch.long)
    continuous = torch.linspace(0, 1, 5).reshape(-1, 1)
    targets = torch.sin(continuous[:, 0])

    reports = [
        gp.fit(
            continuous[:count],
            categorical[:count],
            targets[:count],
            transform_version=count,
        )
        for count in range(2, 6)
    ]

    assert [report.kind for report in reports] == [
        "initial",
        "update",
        "full_refit",
        "update",
    ]
    assert gp.full_refit_count == 2
    assert gp.updates_since_full_refit == 1


def test_disabled_optimizer_reuse_resets_state_on_warm_update() -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
            optimizer="adam",
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
            reuse_optimizer_state=False,
        ),
        runtime=RuntimeConfig(seed=5),
        generator=make_generator("cpu", 5),
    )
    continuous = torch.linspace(0, 1, 4).reshape(-1, 1)
    categorical = torch.empty((4, 0), dtype=torch.long)
    targets = continuous[:, 0].square()
    gp.fit(continuous[:3], categorical[:3], targets[:3], transform_version=1)
    assert gp.optimizer is not None and gp.model is not None
    optimizer_id = id(gp.optimizer)
    model_id = id(gp.model)

    gp.fit(continuous, categorical, targets, transform_version=1)

    assert gp.optimizer is not None and id(gp.optimizer) != optimizer_id
    assert gp.model is not None and id(gp.model) == model_id


def test_psgld_custom_schedule_survives_optimizer_state_round_trip() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PreconditionedSGLD(
        [parameter],
        lr=0.01,
        factor=0.25,
        pretrain_steps=1,
        generator=make_generator("cpu", 8),
    )
    for _ in range(2):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    state = optimizer.state_dict()

    restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    restored = PreconditionedSGLD(
        [restored_parameter],
        lr=0.5,
        factor=1.0,
        pretrain_steps=99,
        generator=make_generator("cpu", 9),
    )
    restored.load_state_dict(state)

    assert restored.steps == 2
    assert restored.pretrain_steps == 1
    assert restored.factor == 0.25
    assert restored.state[restored_parameter]["square_avg"].numel() == 1


def test_model_initialization_is_seeded_without_mutating_global_rng() -> None:
    continuous = torch.linspace(0, 1, 6).reshape(-1, 1)
    categorical = (torch.arange(6) % 3).reshape(-1, 1)
    targets = continuous[:, 0] + categorical[:, 0].float()
    torch.manual_seed(12345)
    global_state = torch.random.get_rng_state().clone()

    first = _surrogate(categories=(3,))
    first.fit(continuous, categorical, targets, transform_version=1)
    assert torch.equal(torch.random.get_rng_state(), global_state)

    second = _surrogate(categories=(3,))
    second.fit(continuous, categorical, targets, transform_version=1)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert first.model is not None and second.model is not None
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])

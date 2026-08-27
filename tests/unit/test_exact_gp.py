# SPDX-License-Identifier: MIT

from __future__ import annotations

import gpytorch  # type: ignore[import-untyped]
import pytest
import torch

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.errors import NumericalError
from leanhebo.gp import ExactGPSurrogate
from leanhebo.gp.kernel import (
    MixedFeatureExtractor,
    build_kernel,
    initialize_base_numeric_lengthscales,
)
from leanhebo.runtime.rng import make_generator


def _surrogate(*, categories: tuple[int, ...] = ()) -> ExactGPSurrogate:
    return ExactGPSurrogate(
        num_continuous=1,
        category_sizes=categories,
        config=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
        ),
        runtime=RuntimeConfig(seed=4),
        generator=make_generator("cpu", 4),
    )


def _current_training_loss(gp: ExactGPSurrogate) -> float:
    assert gp.model is not None and gp.likelihood is not None
    gp.model.train()
    gp.likelihood.train()
    try:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp.model)
        with torch.no_grad():
            return float(gp._training_loss(mll).cpu())
    finally:
        gp.model.eval()
        gp.likelihood.eval()


def test_adam_fits_and_predicts() -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
        ),
        runtime=RuntimeConfig(seed=3),
        generator=make_generator("cpu", 3),
    )
    continuous = torch.tensor([[0.0], [0.4], [0.7], [1.0]])
    categorical = torch.empty((4, 0), dtype=torch.long)
    targets = torch.tensor([1.0, 0.2, 0.1, 0.8])

    report = gp.fit(continuous, categorical, targets, transform_version=1)
    mean, variance, noise = gp.predict(
        torch.tensor([[0.25], [0.85]]),
        torch.empty((2, 0), dtype=torch.long),
    )

    assert report.kind == "initial"
    assert report.completed_steps == 1
    assert report.final_loss == pytest.approx(_current_training_loss(gp))
    assert isinstance(gp.optimizer, torch.optim.Adam)
    assert mean.shape == variance.shape == (2,)
    assert torch.isfinite(mean).all()
    assert torch.all(variance > 0)
    assert noise > 0


def test_gp_fit_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    gp = _surrogate()
    continuous = torch.tensor([[0.0], [0.5], [1.0]])
    categorical = torch.empty((3, 0), dtype=torch.long)
    targets = continuous[:, 0].square()
    calls = 0

    def fail(_: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise RuntimeError("cholesky failed")

    monkeypatch.setattr(gp, "_training_loss", fail)

    with pytest.raises(NumericalError, match="cholesky failed"):
        gp.fit(continuous, categorical, targets, transform_version=1)

    assert calls == 1


def test_nonfinite_loss_is_rejected_before_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = _surrogate()
    continuous = torch.tensor([[0.0], [0.5], [1.0]])
    categorical = torch.empty((3, 0), dtype=torch.long)
    targets = continuous[:, 0].square()

    def nonfinite_loss(_: object) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(gp, "_training_loss", nonfinite_loss)

    with pytest.raises(NumericalError, match="non-finite loss"):
        gp.fit(continuous, categorical, targets, transform_version=1)

    assert gp.optimizer is not None
    assert not gp.optimizer.state


def test_early_stopping_does_not_apply_an_unevaluated_step() -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
            initial_steps=5,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
            early_stopping=True,
            patience=1,
            relative_tolerance=1e30,
        ),
        runtime=RuntimeConfig(seed=4),
        generator=make_generator("cpu", 4),
    )
    continuous = torch.tensor([[0.0], [0.5], [1.0]])
    categorical = torch.empty((3, 0), dtype=torch.long)
    targets = continuous[:, 0].square()

    report = gp.fit(continuous, categorical, targets, transform_version=1)

    assert report.early_stopped
    assert report.completed_steps == 1
    assert report.final_loss == pytest.approx(_current_training_loss(gp))
    assert gp.optimizer is not None
    assert {int(state["step"].item()) for state in gp.optimizer.state.values()} == {1}


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

    torch.testing.assert_close(restored.input_scaler.data_min_, torch.tensor([40.0]))  # type: ignore[union-attr]
    torch.testing.assert_close(restored.input_scaler.data_max_, torch.tensor([65.0]))  # type: ignore[union-attr]
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_categorical_only_gp_uses_checkpointed_identity_input_scaler() -> None:
    gp = ExactGPSurrogate(
        num_continuous=0,
        category_sizes=(2,),
        config=GPConfig(initial_steps=1, update_steps=1),
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


def test_predict_skips_redundant_eval_walks_and_corrects_training_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = _surrogate()
    continuous = torch.linspace(0, 1, 6).reshape(-1, 1)
    categorical = torch.empty((6, 0), dtype=torch.long)
    targets = torch.sin(continuous[:, 0])
    gp.fit(continuous, categorical, targets, transform_version=1)
    assert gp.model is not None and gp.likelihood is not None

    eval_calls: list[torch.nn.Module] = []
    module_eval = torch.nn.Module.eval

    def tracked_eval(module: torch.nn.Module) -> torch.nn.Module:
        eval_calls.append(module)
        return module_eval(module)

    monkeypatch.setattr(torch.nn.Module, "eval", tracked_eval)

    gp.predict(continuous[:2], categorical[:2])
    assert eval_calls == []

    gp.model.train()
    assert gp.model.training and gp.likelihood.training
    gp.predict(continuous[:2], categorical[:2])
    assert eval_calls == [gp.model]
    assert not gp.model.training and not gp.likelihood.training

    eval_calls.clear()
    gp.likelihood.train()
    assert not gp.model.training and gp.likelihood.training
    gp.predict(continuous[:2], categorical[:2])
    assert eval_calls == [gp.likelihood]
    assert not gp.model.training and not gp.likelihood.training


def test_numeric_lengthscale_uses_median_of_all_pairwise_distances() -> None:
    extractor = MixedFeatureExtractor(1, ())
    kernel = build_kernel(num_continuous=1, feature_extractor=extractor, ard=True)
    values = torch.tensor([[0.0], [1.0], [2.0], [100.0]])

    initialize_base_numeric_lengthscales(
        kernel.base_kernel,
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
    assert gp.updates_since_full_refit == 1


def test_default_update_avoids_adam_snapshot_and_full_refit_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=2,
            full_refit_growth_factor=None,
            kernel_initialization_samples=16,
        ),
        runtime=RuntimeConfig(seed=5),
        generator=make_generator("cpu", 5),
    )
    continuous = torch.linspace(0, 1, 5).reshape(-1, 1)
    categorical = torch.empty((5, 0), dtype=torch.long)
    targets = torch.sin(continuous[:, 0])

    gp.fit(continuous[:3], categorical[:3], targets[:3], transform_version=1)
    assert gp.optimizer is not None
    optimizer = gp.optimizer

    def unexpected_snapshot() -> dict[str, object]:
        raise AssertionError("default updates and full refits must not snapshot Adam state")

    monkeypatch.setattr(optimizer, "state_dict", unexpected_snapshot)
    gp.fit(continuous[:4], categorical[:4], targets[:4], transform_version=2)

    assert gp.optimizer is optimizer
    assert gp.optimizer.param_groups[0]["betas"] == (0.9, 0.99)
    assert {int(state["step"].item()) for state in gp.optimizer.state.values()} == {2}

    report = gp.fit(continuous, categorical, targets, transform_version=3)

    assert report.kind == "full_refit"
    assert gp.optimizer is not None and gp.optimizer is not optimizer
    assert {int(state["step"].item()) for state in gp.optimizer.state.values()} == {1}


def test_reconstructed_warm_update_transfers_adam_state() -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
            use_set_train_data=False,
            kernel_initialization_samples=16,
        ),
        runtime=RuntimeConfig(seed=5),
        generator=make_generator("cpu", 5),
    )
    continuous = torch.linspace(0, 1, 4).reshape(-1, 1)
    categorical = torch.empty((4, 0), dtype=torch.long)
    targets = continuous[:, 0].square()
    gp.fit(continuous[:3], categorical[:3], targets[:3], transform_version=1)
    assert gp.optimizer is not None
    optimizer = gp.optimizer

    gp.fit(continuous, categorical, targets, transform_version=1)

    assert gp.optimizer is not None and gp.optimizer is not optimizer
    assert {int(state["step"].item()) for state in gp.optimizer.state.values()} == {2}


def test_disabled_optimizer_reuse_resets_state_on_warm_update() -> None:
    gp = ExactGPSurrogate(
        num_continuous=1,
        category_sizes=(),
        config=GPConfig(
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

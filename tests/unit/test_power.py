# SPDX-License-Identifier: MIT
"""Tests for power transforms and objective-transform state."""

from __future__ import annotations

import pytest
import torch
from sklearn.preprocessing import PowerTransformer as SklearnPowerTransformer

from leanhebo.config import WarpConfig
from leanhebo.transforms import (
    OutputTransform,
    PowerTransformDomainError,
    PowerTransformer,
    PowerTransformFitError,
    box_cox,
    fit_power_lambda,
    inverse_box_cox,
    inverse_yeo_johnson,
    yeo_johnson,
)


@pytest.mark.parametrize("lmbda", [-2.0, 0.0, 0.5, 1.0, 2.0])
def test_box_cox_inverse_round_trip(lmbda: float) -> None:
    x = torch.logspace(-2, 2, 101, dtype=torch.float64)
    restored = inverse_box_cox(box_cox(x, lmbda), lmbda)
    torch.testing.assert_close(restored, x, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("lmbda", [-2.0, 0.0, 0.5, 1.0, 2.0, 4.0])
def test_yeo_johnson_inverse_round_trip(lmbda: float) -> None:
    x = torch.linspace(-20, 20, 201, dtype=torch.float64)
    restored = inverse_yeo_johnson(yeo_johnson(x, lmbda), lmbda)
    torch.testing.assert_close(restored, x, atol=1e-9, rtol=1e-8)


def test_formula_limit_branches() -> None:
    positive = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
    signed = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(box_cox(positive, 0.0), positive.log())
    torch.testing.assert_close(box_cox(positive, 1.0), positive - 1)
    torch.testing.assert_close(
        yeo_johnson(signed, 1.0),
        signed,
    )
    negative = signed[signed < 0]
    torch.testing.assert_close(yeo_johnson(negative, 2.0), -torch.log1p(-negative))


@pytest.mark.parametrize("lmbda", [-3.0, 0.0, 1.0, 2.0, 4.0])
def test_yeo_johnson_round_trip_has_finite_gradients(lmbda: float) -> None:
    x = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float64, requires_grad=True)
    restored = inverse_yeo_johnson(yeo_johnson(x, lmbda), lmbda)
    restored.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize(
    ("method", "x"),
    [
        ("box-cox", torch.tensor([0.2, 0.4, 1.0, 1.5, 3.0, 9.0], dtype=torch.float64)),
        (
            "yeo-johnson",
            torch.tensor([-4.0, -2.0, -0.3, 0.0, 0.5, 2.0, 8.0], dtype=torch.float64),
        ),
    ],
)
def test_fitted_lambda_matches_sklearn(method: str, x: torch.Tensor) -> None:
    expected = SklearnPowerTransformer(method=method, standardize=False).fit(x[:, None].numpy())
    actual = fit_power_lambda(x, method)  # type: ignore[arg-type]
    assert actual.item() == pytest.approx(expected.lambdas_[0], abs=2e-5)


def test_power_transformer_standardizes_and_serializes() -> None:
    x = torch.tensor([-4.0, -1.0, 0.0, 0.5, 3.0, 9.0], dtype=torch.float64)
    transformer = PowerTransformer("yeo-johnson").fit(x)
    transformed = transformer.transform(x)
    assert transformed.mean().item() == pytest.approx(0.0, abs=1e-12)
    assert transformed.std(correction=0).item() == pytest.approx(1.0, abs=1e-12)
    torch.testing.assert_close(transformer.inverse_transform(transformed), x)

    restored = PowerTransformer("box-cox")
    restored.load_state_dict(transformer.state_dict())
    assert restored.method == "yeo-johnson"
    assert restored.lambda_.dtype == torch.float64
    torch.testing.assert_close(restored.transform(x), transformed)


def test_output_transform_accepts_warp_config_and_selects_box_cox() -> None:
    x = torch.tensor([[1.0], [2.0], [4.0], [8.0]], dtype=torch.float64)
    transform = OutputTransform(WarpConfig()).fit(x)
    transformed = transform.transform(x)

    assert transform.method == "box-cox"
    assert transform.requested_method == "auto"
    assert transform.version == 1
    assert transform.output_standardizer_version == 1
    assert transform.output_warp_version == 1
    assert transformed.shape == x.shape
    torch.testing.assert_close(transform.inverse_transform(transformed), x, atol=1e-9, rtol=1e-9)


def test_output_transform_auto_uses_yeo_johnson_for_signed_values() -> None:
    x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 8.0], dtype=torch.float32)
    transform = OutputTransform().fit(x)
    assert transform.method == "yeo-johnson"
    assert transform.transform(x).dtype == x.dtype
    torch.testing.assert_close(transform.inverse_transform(transform.transform(x)), x)


def test_output_transform_auto_falls_back_for_constant_and_short_data() -> None:
    constant = OutputTransform().fit(torch.ones(5, dtype=torch.float64))
    short = OutputTransform().fit(torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert constant.method == "none"
    assert short.method == "none"
    assert torch.equal(
        constant.transform(torch.ones(5, dtype=torch.float64)),
        torch.zeros(5, dtype=torch.float64),
    )


def test_output_transform_preserves_nonfinite_entries() -> None:
    x = torch.tensor([1.0, 2.0, 4.0, torch.nan, torch.inf], dtype=torch.float64)
    transform = OutputTransform().fit(x)
    transformed = transform.transform(x)
    restored = transform.inverse_transform(transformed)

    assert torch.isnan(transformed[-2]) and torch.isposinf(transformed[-1])
    assert torch.isnan(restored[-2]) and torch.isposinf(restored[-1])
    torch.testing.assert_close(restored[:3], x[:3])


def test_output_transform_state_restores_method_config_versions_and_dtype() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 4.0], dtype=torch.float64)
    transform = OutputTransform(method="auto", refit_interval=None).fit(x)
    state = transform.state_dict()
    restored = OutputTransform(method="none")
    restored.load_state_dict(state)

    assert restored.method == transform.method
    assert restored.requested_method == "auto"
    assert restored.refit_interval is None
    assert restored.version == transform.version
    assert restored.lambda_.dtype == torch.float64
    torch.testing.assert_close(restored.transform(x), transform.transform(x))


def test_output_transform_refit_schedule_and_force() -> None:
    first = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    second = torch.tensor([1.0, 3.0, 9.0, 27.0], dtype=torch.float64)
    transform = OutputTransform(refit_interval=3).fit(first)
    original_lambda = transform.lambda_.clone()
    transform.fit(second)
    assert transform.version == 1
    transform.fit(second)
    assert transform.version == 1
    transform.fit(second, force=True)
    assert transform.version == 2
    assert not torch.equal(transform.lambda_, original_lambda)


def test_explicit_power_transform_reports_domain_and_fit_errors() -> None:
    with pytest.raises(PowerTransformDomainError):
        OutputTransform(method="box-cox").fit(torch.tensor([-1.0, 1.0, 2.0]))
    with pytest.raises(PowerTransformFitError):
        PowerTransformer("yeo-johnson").fit(torch.ones(4))


def test_inverse_domain_errors_are_typed() -> None:
    with pytest.raises(PowerTransformDomainError):
        inverse_box_cox(torch.tensor([-2.0]), 1.0)


def test_output_transform_requires_scalar_objective_shape() -> None:
    with pytest.raises(ValueError, match="objective shape"):
        OutputTransform().fit(torch.ones(3, 2))

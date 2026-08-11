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
    PowerTransformFitError,
)


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
def test_output_transform_matches_sklearn(method: str, x: torch.Tensor) -> None:
    expected = SklearnPowerTransformer(method=method).fit_transform(x[:, None].numpy())[:, 0]
    config = WarpConfig(method=method, standardize_before_warp=False)  # type: ignore[arg-type]
    actual = OutputTransform(config).fit_transform(x)

    torch.testing.assert_close(
        actual,
        torch.from_numpy(expected),
        atol=3e-6,
        rtol=3e-6,
    )


def test_output_transform_accepts_warp_config_and_selects_box_cox() -> None:
    x = torch.tensor([[1.0], [2.0], [4.0], [8.0]], dtype=torch.float64)
    transform = OutputTransform(WarpConfig()).fit(x)
    transformed = transform.transform(x)

    assert transform.method == "box-cox"
    assert transform.version == 1
    assert transformed.shape == x.shape


def test_output_transform_auto_uses_yeo_johnson_for_signed_values() -> None:
    x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 8.0], dtype=torch.float32)
    transform = OutputTransform().fit(x)
    assert transform.method == "yeo-johnson"
    assert transform.transform(x).dtype == x.dtype


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

    assert torch.isnan(transformed[-2]) and torch.isposinf(transformed[-1])


def test_output_transform_state_restores_fitted_transform() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 4.0], dtype=torch.float64)
    config = WarpConfig(method="auto", refit_interval=None)
    transform = OutputTransform(config).fit(x)
    state = transform.state_dict()
    restored = OutputTransform(config).to(dtype=x.dtype)
    restored.load_state_dict(state)

    assert restored.method == transform.method
    assert restored.version == transform.version
    torch.testing.assert_close(restored.transform(x), transform.transform(x))


def test_output_transform_refit_schedule_and_force() -> None:
    first = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    second = torch.tensor([1.0, 3.0, 9.0, 27.0], dtype=torch.float64)
    transform = OutputTransform(WarpConfig(refit_interval=3)).fit(first)
    original = transform.transform(second)
    transform.fit(second)
    assert transform.version == 1
    transform.fit(second)
    assert transform.version == 1
    transform.fit(second, force=True)
    assert transform.version == 2
    assert not torch.equal(transform.transform(second), original)


def test_explicit_power_transform_reports_domain_and_fit_errors() -> None:
    with pytest.raises(PowerTransformDomainError):
        OutputTransform(WarpConfig(method="box-cox")).fit(torch.tensor([-1.0, 1.0, 2.0]))
    with pytest.raises(PowerTransformFitError):
        OutputTransform(WarpConfig(method="yeo-johnson")).fit(torch.ones(4))


def test_output_transform_requires_scalar_objective_shape() -> None:
    with pytest.raises(ValueError, match="objective shape"):
        OutputTransform().fit(torch.ones(3, 2))

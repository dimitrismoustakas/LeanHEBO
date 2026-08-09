# SPDX-License-Identifier: MIT
"""Tests for the pure-Torch feature scalers."""

from __future__ import annotations

import pytest
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from leanhebo.transforms import IdentityScaler, TorchMinMaxScaler, TorchStandardScaler


@pytest.mark.parametrize(
    "scaler",
    [IdentityScaler(), TorchStandardScaler(), TorchMinMaxScaler((-3.0, 7.0))],
)
def test_scaler_inverse_round_trip_and_gradients(scaler: torch.nn.Module) -> None:
    x = (torch.randn(16, 3, dtype=torch.float64) * 4 + 2).requires_grad_()
    scaler.fit(x)
    transformed = scaler.transform(x)
    restored = scaler.inverse_transform(transformed)

    torch.testing.assert_close(restored, x)
    transformed.sum().backward()
    assert x.grad is not None
    assert all(buffer.grad is None for buffer in scaler.buffers())


def test_standard_scaler_matches_sklearn() -> None:
    x = torch.tensor(
        [[1.0, 10.0, 4.0], [3.0, 20.0, 4.0], [8.0, 30.0, 4.0]],
        dtype=torch.float64,
    )
    actual = TorchStandardScaler().fit(x).transform(x)
    expected = StandardScaler().fit_transform(x.numpy())

    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-12, rtol=1e-12)


def test_standard_scaler_ignores_nonfinite_values_and_handles_empty_column() -> None:
    x = torch.tensor(
        [[1.0, 3.0, torch.nan], [2.0, torch.inf, torch.nan], [3.0, 3.0, -torch.inf]],
        dtype=torch.float64,
    )
    scaler = TorchStandardScaler().fit(x)

    torch.testing.assert_close(scaler.mean, torch.tensor([2.0, 3.0, 0.0], dtype=x.dtype))
    torch.testing.assert_close(scaler.std, torch.tensor([(2 / 3) ** 0.5, 1.0, 1.0], dtype=x.dtype))
    assert torch.isnan(scaler.transform(x)[0, 2])
    assert torch.isposinf(scaler.transform(x)[1, 1])
    assert torch.isneginf(scaler.transform(x)[2, 2])


def test_minmax_scaler_matches_sklearn_including_constant_column() -> None:
    x = torch.tensor(
        [[1.0, 10.0, 4.0], [3.0, 20.0, 4.0], [8.0, 30.0, 4.0]],
        dtype=torch.float64,
    )
    scaler = TorchMinMaxScaler((-2.0, 5.0)).fit(x)
    expected = MinMaxScaler(feature_range=(-2.0, 5.0)).fit_transform(x.numpy())

    torch.testing.assert_close(
        scaler.transform(x), torch.from_numpy(expected), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(scaler.inverse_transform(scaler.transform(x)), x)
    assert torch.equal(scaler.transform(x)[:, 2], torch.full((3,), -2.0, dtype=x.dtype))


def test_minmax_all_nonfinite_column_is_neutral() -> None:
    x = torch.tensor([[1.0, torch.nan], [2.0, torch.inf]], dtype=torch.float64)
    scaler = TorchMinMaxScaler().fit(x)
    future = torch.tensor([[1.5, 42.0]], dtype=torch.float64)

    assert scaler.finite_count_[1] == 0
    torch.testing.assert_close(scaler.transform(future)[0, 1], future[0, 1])
    torch.testing.assert_close(scaler.inverse_transform(scaler.transform(future)), future)


@pytest.mark.parametrize("scaler", [TorchStandardScaler(), TorchMinMaxScaler((-1.0, 1.0))])
def test_scaler_state_loads_into_fresh_instance_with_original_dtype(
    scaler: TorchStandardScaler | TorchMinMaxScaler,
) -> None:
    x = torch.tensor([[1.0, 4.0], [3.0, 9.0]], dtype=torch.float64)
    scaler.fit(x)
    state = scaler.state_dict()
    restored = type(scaler)()
    restored.load_state_dict(state)

    assert next(restored.buffers()).dtype == torch.float64
    torch.testing.assert_close(restored.transform(x), scaler.transform(x))


def test_unfitted_scaler_and_bad_feature_count_raise() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        TorchStandardScaler().transform(torch.ones(2, 1))
    scaler = TorchMinMaxScaler().fit(torch.ones(2, 2))
    with pytest.raises(ValueError, match="last dimension"):
        scaler.transform(torch.ones(2, 1))


def test_scaler_buffers_follow_dtype_conversion() -> None:
    scaler = TorchStandardScaler().fit(torch.randn(8, 2)).to(dtype=torch.float64)
    assert scaler.version == 1
    assert scaler.input_scaler_version == 1
    assert scaler.mean.dtype == torch.float64
    assert scaler.std.dtype == torch.float64
    assert scaler.transform(torch.randn(3, 2, dtype=torch.float64)).dtype == torch.float64


def test_minmax_range_keyword_compatibility() -> None:
    scaler = TorchMinMaxScaler(range=(-4.0, 2.0))
    assert scaler.feature_range == (-4.0, 2.0)
    with pytest.raises(ValueError, match="either"):
        TorchMinMaxScaler((-1.0, 1.0), range=(-2.0, 2.0))

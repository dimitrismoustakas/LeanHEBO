# SPDX-License-Identifier: MIT
"""Tests for the pure-Torch feature scalers."""

from __future__ import annotations

import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from leanhebo.transforms import IdentityScaler, TorchMinMaxScaler


def test_scalers_preserve_gradients() -> None:
    x = (torch.randn(16, 3, dtype=torch.float64) * 4 + 2).requires_grad_()
    identity = IdentityScaler().fit(x)
    scaler = TorchMinMaxScaler((-3.0, 7.0)).fit(x)

    torch.testing.assert_close(identity.transform(x), x)
    transformed = scaler.transform(x)
    transformed.sum().backward()
    assert x.grad is not None
    assert all(buffer.grad is None for buffer in scaler.buffers())


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
    assert torch.equal(scaler.transform(x)[:, 2], torch.full((3,), -2.0, dtype=x.dtype))


def test_minmax_all_nonfinite_column_is_neutral() -> None:
    x = torch.tensor([[1.0, torch.nan], [2.0, torch.inf]], dtype=torch.float64)
    scaler = TorchMinMaxScaler().fit(x)
    future = torch.tensor([[1.5, 42.0]], dtype=torch.float64)

    torch.testing.assert_close(scaler.transform(future)[0, 1], future[0, 1])


def test_minmax_state_loads_into_fresh_instance_with_original_dtype() -> None:
    x = torch.tensor([[1.0, 4.0], [3.0, 9.0]], dtype=torch.float64)
    scaler = TorchMinMaxScaler((-1.0, 1.0)).fit(x)
    state = scaler.state_dict()
    restored = TorchMinMaxScaler((-1.0, 1.0))
    restored.load_state_dict(state)

    assert next(restored.buffers()).dtype == torch.float64
    torch.testing.assert_close(restored.transform(x), scaler.transform(x))


def test_unfitted_scaler_and_bad_feature_count_raise() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        TorchMinMaxScaler().transform(torch.ones(2, 1))
    scaler = TorchMinMaxScaler().fit(torch.ones(2, 2))
    with pytest.raises(ValueError, match="last dimension"):
        scaler.transform(torch.ones(2, 1))


def test_minmax_buffers_follow_dtype_conversion() -> None:
    scaler = TorchMinMaxScaler().fit(torch.randn(8, 2)).to(dtype=torch.float64)
    assert scaler.data_min_.dtype == torch.float64
    assert scaler.scale_.dtype == torch.float64
    assert scaler.transform(torch.randn(3, 2, dtype=torch.float64)).dtype == torch.float64

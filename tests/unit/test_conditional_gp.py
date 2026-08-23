# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import gpytorch
import pytest
import torch
from gpytorch.lazy import LazyEvaluatedKernelTensor
from linear_operator.operators import DenseLinearOperator

from leanhebo.config import GPConfig, RuntimeConfig
from leanhebo.data import EncodedBatch
from leanhebo.gp.conditional import ConditionalExactGPSurrogate
from leanhebo.gp.conditional_kernel import (
    ActivityFactorizedProductKernel,
    ActivityGroupSpec,
    ConditionalKernelLayout,
)
from leanhebo.space import Any, Bool, Categorical, CompiledSpace, Eq, Float


def _branch_space() -> CompiledSpace:
    return CompiledSpace(
        (
            Categorical("kind", ("plain", "branch")),
            Float("root", 0.0, 1.0),
            Float("child", 0.0, 1.0, active_when=Eq("kind", "branch")),
        ),
        dtype=torch.float64,
    )


def _surrogate(
    space: CompiledSpace,
    *,
    fantasy: bool = False,
    seed: int = 7,
) -> ConditionalExactGPSurrogate:
    return ConditionalExactGPSurrogate(
        space=space,
        config=GPConfig(
            initial_steps=2,
            update_steps=0 if fantasy else 1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
            use_fantasy_updates=fantasy,
        ),
        runtime=RuntimeConfig(dtype="float64", seed=seed),
        generator=torch.Generator().manual_seed(seed),
    )


def test_layout_requires_an_exact_partition() -> None:
    layout = ConditionalKernelLayout(
        root_continuous_indices=(0,),
        root_categorical_indices=(),
        groups=(ActivityGroupSpec((0,), ()),),
    )

    with pytest.raises(ValueError, match="exact partition"):
        layout.validate(num_continuous=2, num_categorical=0)


def test_conditional_gp_eagerly_evaluates_its_dense_kernel() -> None:
    surrogate = _surrogate(_branch_space())
    layout = ConditionalKernelLayout((0,), (), (ActivityGroupSpec((1,), ()),))
    kernel = ActivityFactorizedProductKernel(category_sizes=(), layout=layout, ard=True)
    continuous = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    categorical = torch.empty((2, 0), dtype=torch.int64)
    activity = torch.tensor([[False], [True]])
    packed = kernel.pack_inputs(continuous, categorical, activity)

    with gpytorch.settings.lazily_evaluate_kernels(True):
        assert isinstance(kernel(packed, packed), LazyEvaluatedKernelTensor)
        with surrogate._settings():
            assert isinstance(kernel(packed, packed), DenseLinearOperator)
        assert gpytorch.settings.lazily_evaluate_kernels.on()


def test_activity_kernel_batch_shape_does_not_traverse_feature_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ConditionalKernelLayout((0,), (), (ActivityGroupSpec((1,), ()),))
    kernel = ActivityFactorizedProductKernel(category_sizes=(), layout=layout, ard=True)

    def unexpected_traversal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("batch shape traversed the fixed unbatched feature blocks")

    monkeypatch.setattr(gpytorch.kernels.Kernel, "sub_kernels", unexpected_traversal)
    assert kernel.batch_shape == torch.Size()


def test_activity_factor_has_exact_unit_diagonal_and_is_psd() -> None:
    layout = ConditionalKernelLayout(
        root_continuous_indices=(0,),
        root_categorical_indices=(),
        groups=(ActivityGroupSpec((1,), ()), ActivityGroupSpec()),
    )
    layout.validate(num_continuous=2, num_categorical=0)
    kernel = ActivityFactorizedProductKernel(
        category_sizes=(),
        layout=layout,
        ard=True,
    ).double()
    continuous = torch.tensor(
        [[0.0, 0.1], [0.2, 0.9], [0.4, 0.3], [0.7, 0.6], [1.0, 0.8]],
        dtype=torch.float64,
    )
    categorical = torch.empty((5, 0), dtype=torch.int64)
    activity = torch.tensor(
        [[False, False], [False, True], [True, False], [True, True], [True, False]]
    )
    packed = kernel.pack_inputs(continuous, categorical, activity)
    gram = kernel(packed, packed).to_dense()

    assert torch.equal(kernel.alpha + kernel.beta, torch.ones_like(kernel.alpha))
    assert torch.equal(gram.diagonal(), torch.ones(5, dtype=torch.float64))
    assert float(torch.linalg.eigvalsh(gram).min().detach()) >= -1e-12


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 2e-5), (torch.float64, 1e-12)],
)
@pytest.mark.parametrize("seed", range(4))
def test_activity_factor_is_psd_across_random_group_layouts(
    dtype: torch.dtype,
    tolerance: float,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    layout = ConditionalKernelLayout(
        root_continuous_indices=(0,),
        root_categorical_indices=(),
        groups=(
            ActivityGroupSpec((1,), ()),
            ActivityGroupSpec((), (0,)),
            ActivityGroupSpec(),
        ),
    )
    layout.validate(num_continuous=2, num_categorical=1)
    kernel = ActivityFactorizedProductKernel(
        category_sizes=(3,),
        layout=layout,
        ard=True,
    ).to(dtype=dtype)
    continuous = torch.rand((18, 2), dtype=dtype, generator=generator)
    categorical = torch.randint(0, 3, (18, 1), generator=generator)
    activity = torch.rand((18, 3), generator=generator) > 0.45
    packed = kernel.pack_inputs(continuous, categorical, activity)

    gram = kernel(packed, packed).to_dense()

    torch.testing.assert_close(gram, gram.T, rtol=0.0, atol=tolerance)
    torch.testing.assert_close(
        gram.diagonal(),
        torch.ones(18, dtype=dtype),
        rtol=0.0,
        atol=tolerance,
    )
    assert float(torch.linalg.eigvalsh(gram).min().detach()) >= -tolerance


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dense_activity_factors_match_cpu_active_blocks() -> None:
    layout = ConditionalKernelLayout(
        root_continuous_indices=(0,),
        root_categorical_indices=(),
        groups=(ActivityGroupSpec((1,), ()), ActivityGroupSpec((), (0,))),
    )
    layout.validate(num_continuous=2, num_categorical=1)
    cpu = ActivityFactorizedProductKernel(
        category_sizes=(3,),
        layout=layout,
        ard=True,
    ).double()
    cuda = (
        ActivityFactorizedProductKernel(
            category_sizes=(3,),
            layout=layout,
            ard=True,
        )
        .double()
        .cuda()
    )
    cuda.load_state_dict(cpu.state_dict())
    continuous = torch.tensor(
        [[0.0, 0.1], [0.2, 0.9], [0.4, 0.3], [0.7, 0.6], [1.0, 0.8]],
        dtype=torch.float64,
    )
    categorical = torch.tensor([[0], [1], [2], [1], [0]], dtype=torch.int64)
    activity = torch.tensor(
        [[False, False], [False, True], [True, False], [True, True], [True, False]]
    )
    cpu_packed = cpu.pack_inputs(continuous, categorical, activity)
    cuda_packed = cuda.pack_inputs(
        continuous.cuda(),
        categorical.cuda(),
        activity.cuda(),
    )

    torch.testing.assert_close(
        cuda(cuda_packed, cuda_packed).to_dense().cpu(),
        cpu(cpu_packed, cpu_packed).to_dense(),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        cuda(cuda_packed, cuda_packed, diag=True).cpu(),
        cpu(cpu_packed, cpu_packed, diag=True),
        rtol=0.0,
        atol=0.0,
    )


def test_zero_feature_root_and_group_are_constant_identities() -> None:
    layout = ConditionalKernelLayout((), (), (ActivityGroupSpec(),))
    layout.validate(num_continuous=0, num_categorical=0)
    kernel = ActivityFactorizedProductKernel(
        category_sizes=(),
        layout=layout,
        ard=True,
    ).double()
    continuous = torch.empty((2, 0), dtype=torch.float64)
    categorical = torch.empty((2, 0), dtype=torch.int64)
    activity = torch.tensor([[False], [True]])
    packed = kernel.pack_inputs(continuous, categorical, activity)
    gram = kernel(packed, packed).to_dense()

    rho = kernel.rho.detach()[0]
    expected = torch.stack(
        (
            torch.stack((rho.new_tensor(1.0), rho)),
            torch.stack((rho, rho.new_tensor(1.0))),
        )
    )
    torch.testing.assert_close(gram, expected, rtol=0.0, atol=0.0)


def test_inactive_local_values_do_not_change_kernel_rows() -> None:
    layout = ConditionalKernelLayout((0,), (), (ActivityGroupSpec((1,), ()),))
    kernel = ActivityFactorizedProductKernel(
        category_sizes=(),
        layout=layout,
        ard=True,
    ).double()
    categorical = torch.empty((3, 0), dtype=torch.int64)
    activity = torch.tensor([[False], [False], [True]])
    first = torch.tensor([[0.2, 0.1], [0.2, 0.9], [0.8, 0.4]], dtype=torch.float64)
    packed = kernel.pack_inputs(first, categorical, activity)
    gram = kernel(packed, packed).to_dense()

    torch.testing.assert_close(gram[0], gram[1], rtol=0.0, atol=0.0)


def test_activity_logits_have_one_stable_parameterization() -> None:
    layout = ConditionalKernelLayout((), (), (ActivityGroupSpec(), ActivityGroupSpec()))
    kernel = ActivityFactorizedProductKernel(
        category_sizes=(),
        layout=layout,
        ard=True,
    ).double()

    torch.testing.assert_close(kernel.alpha, torch.full((2,), 0.2, dtype=torch.float64))
    torch.testing.assert_close(kernel.beta, torch.full((2,), 0.8, dtype=torch.float64))
    torch.testing.assert_close(kernel.rho, torch.full((2,), math.sqrt(0.8), dtype=torch.float64))
    assert {name for name, *_ in kernel.named_priors()} == {"activity_logit_prior"}

    with torch.no_grad():
        kernel.activity_logit.copy_(torch.tensor([-20.0, 20.0], dtype=torch.float64))
    continuous = torch.empty((3, 0), dtype=torch.float64)
    categorical = torch.empty((3, 0), dtype=torch.int64)
    activity = torch.tensor([[False, False], [True, False], [True, True]])
    packed = kernel.pack_inputs(continuous, categorical, activity)
    gram = kernel(packed, packed).to_dense()
    gram.sum().backward()

    assert torch.isfinite(gram).all()
    assert kernel.activity_logit.grad is not None
    assert torch.isfinite(kernel.activity_logit.grad).all()


def test_conditional_surrogate_derives_activity_and_masks_scaling() -> None:
    space = _branch_space()
    surrogate = _surrogate(space)
    encoded = EncodedBatch(
        torch.tensor(
            [[0.0, 0.1], [0.4, 0.9], [0.6, 0.2], [1.0, 0.8]],
            dtype=torch.float64,
        ),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.int64),
    )
    targets = torch.tensor([1.0, 1.2, 0.3, 0.1], dtype=torch.float64)

    surrogate.fit(encoded.continuous, encoded.categorical, targets, transform_version=0)

    assert surrogate.train_activity is not None
    assert surrogate.train_activity[:, 0].tolist() == [False, False, True, True]
    assert surrogate.train_continuous is not None
    assert torch.equal(surrogate.train_continuous[:2, 1], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        surrogate.masked_input_scaler.data_min_,
        torch.tensor([0.0, 0.2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        surrogate.masked_input_scaler.data_max_,
        torch.tensor([1.0, 0.8], dtype=torch.float64),
    )

    inactive = EncodedBatch(
        torch.tensor([[0.3, 0.1], [0.3, 0.9]], dtype=torch.float64),
        torch.zeros((2, 1), dtype=torch.int64),
    )
    mean, variance, _ = surrogate.predict(inactive.continuous, inactive.categorical)
    torch.testing.assert_close(mean[0], mean[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(variance[0], variance[1], rtol=0.0, atol=0.0)

    assert surrogate.model is not None
    scales = [
        module
        for module in surrogate.model.modules()
        if isinstance(module, gpytorch.kernels.ScaleKernel)
    ]
    assert len(scales) == 1
    model = surrogate.model
    activity_kernel = model.covar_module.base_kernel
    assert isinstance(activity_kernel, ActivityFactorizedProductKernel)
    packed = activity_kernel.pack_inputs(
        surrogate.train_continuous,
        surrogate.train_categorical,
        surrogate.train_activity,
    )
    covariance = model.covar_module(packed).to_dense()
    expected_diagonal = model.covar_module.outputscale.expand(4)
    torch.testing.assert_close(covariance.diagonal(), expected_diagonal, rtol=0.0, atol=0.0)


def test_unobserved_group_uses_compiled_bounds_and_predicts_finitely() -> None:
    space = _branch_space()
    surrogate = _surrogate(space)
    encoded = EncodedBatch(
        torch.tensor([[0.0, 0.2], [0.5, 0.8], [1.0, 0.4]], dtype=torch.float64),
        torch.zeros((3, 1), dtype=torch.int64),
    )
    targets = torch.tensor([0.0, 0.2, 0.4], dtype=torch.float64)

    surrogate.fit(encoded.continuous, encoded.categorical, targets, transform_version=0)

    assert surrogate.masked_input_scaler.active_count_.tolist() == [3, 0]
    assert surrogate.masked_input_scaler.data_min_.tolist() == [0.0, 0.0]
    assert surrogate.masked_input_scaler.data_max_.tolist() == [1.0, 1.0]
    active = EncodedBatch(
        torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.int64),
    )
    mean, variance, _ = surrogate.predict(active.continuous, active.categorical)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()


def test_conditional_state_round_trip_reproduces_posterior() -> None:
    space = _branch_space()
    surrogate = _surrogate(space, seed=11)
    encoded = space.encode(
        [
            {"kind": "plain", "root": 0.0},
            {"kind": "plain", "root": 1.0},
            {"kind": "branch", "root": 0.3, "child": 0.2},
            {"kind": "branch", "root": 0.7, "child": 0.8},
        ]
    )
    targets = torch.tensor([1.0, 1.2, 0.4, 0.1], dtype=torch.float64)
    surrogate.fit(encoded.continuous, encoded.categorical, targets, transform_version=4)
    expected = surrogate.predict(encoded.continuous, encoded.categorical)

    restored = _surrogate(space, seed=99)
    restored.load_state_dict(surrogate.state_dict())
    actual = restored.predict(encoded.continuous, encoded.categorical)

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    assert restored.train_activity is not None
    assert surrogate.train_activity is not None
    assert torch.equal(restored.train_activity, surrogate.train_activity)


def test_conditional_default_update_avoids_adam_snapshot_and_full_refit_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _branch_space()
    surrogate = ConditionalExactGPSurrogate(
        space=space,
        config=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=2,
            full_refit_growth_factor=None,
        ),
        runtime=RuntimeConfig(dtype="float64", seed=5),
        generator=torch.Generator().manual_seed(5),
    )
    encoded = space.encode(
        [
            {"kind": "plain", "root": 0.0},
            {"kind": "plain", "root": 1.0},
            {"kind": "branch", "root": 0.2, "child": 0.1},
            {"kind": "branch", "root": 0.5, "child": 0.4},
            {"kind": "branch", "root": 0.8, "child": 0.9},
        ]
    )
    targets = torch.tensor([1.0, 1.2, 0.8, 0.4, 0.1], dtype=torch.float64)

    surrogate.fit(
        encoded.continuous[:3],
        encoded.categorical[:3],
        targets[:3],
        transform_version=1,
    )
    assert surrogate.optimizer is not None
    optimizer = surrogate.optimizer

    def unexpected_snapshot() -> dict[str, object]:
        raise AssertionError("default updates and full refits must not snapshot Adam state")

    monkeypatch.setattr(optimizer, "state_dict", unexpected_snapshot)
    surrogate.fit(
        encoded.continuous[:4],
        encoded.categorical[:4],
        targets[:4],
        transform_version=2,
    )

    assert surrogate.optimizer is optimizer
    assert surrogate.optimizer.param_groups[0]["betas"] == (0.9, 0.99)
    assert {int(state["step"].item()) for state in surrogate.optimizer.state.values()} == {2}

    report = surrogate.fit(
        encoded.continuous,
        encoded.categorical,
        targets,
        transform_version=3,
    )

    assert report.kind == "full_refit"
    assert surrogate.optimizer is not None and surrogate.optimizer is not optimizer
    actual_steps = {int(state["step"].item()) for state in surrogate.optimizer.state.values()}
    assert actual_steps == {1}


def test_conditional_reconstructed_warm_update_transfers_adam_state() -> None:
    space = _branch_space()
    surrogate = ConditionalExactGPSurrogate(
        space=space,
        config=GPConfig(
            initial_steps=1,
            update_steps=1,
            full_refit_interval=None,
            full_refit_growth_factor=None,
            use_set_train_data=False,
        ),
        runtime=RuntimeConfig(dtype="float64", seed=5),
        generator=torch.Generator().manual_seed(5),
    )
    encoded = space.encode(
        [
            {"kind": "plain", "root": 0.0},
            {"kind": "plain", "root": 1.0},
            {"kind": "branch", "root": 0.2, "child": 0.1},
            {"kind": "branch", "root": 0.8, "child": 0.9},
        ]
    )
    targets = torch.tensor([1.0, 1.2, 0.8, 0.1], dtype=torch.float64)
    surrogate.fit(
        encoded.continuous[:3],
        encoded.categorical[:3],
        targets[:3],
        transform_version=1,
    )
    assert surrogate.optimizer is not None
    optimizer = surrogate.optimizer

    surrogate.fit(
        encoded.continuous,
        encoded.categorical,
        targets,
        transform_version=1,
    )

    assert surrogate.optimizer is not None and surrogate.optimizer is not optimizer
    actual_steps = {int(state["step"].item()) for state in surrogate.optimizer.state.values()}
    assert actual_steps == {2}


def test_any_condition_ignores_an_inactive_alternative_parent() -> None:
    space = CompiledSpace(
        (
            Categorical("mode", ("a", "b")),
            Bool("a_enabled", active_when=Eq("mode", "a")),
            Bool("b_enabled", active_when=Eq("mode", "b")),
            Float(
                "child",
                0.0,
                1.0,
                active_when=Any(Eq("a_enabled", True), Eq("b_enabled", True)),
            ),
        ),
        dtype=torch.float64,
    )
    surrogate = _surrogate(space)
    encoded = space.encode(
        [
            {"mode": "a", "a_enabled": True, "child": 0.2},
            {"mode": "a", "a_enabled": False},
            {"mode": "b", "b_enabled": True, "child": 0.4},
            {"mode": "b", "b_enabled": False},
        ]
    )
    surrogate.fit(
        encoded.continuous,
        encoded.categorical,
        torch.tensor([0.2, 0.8, 0.4, 1.0], dtype=torch.float64),
        transform_version=0,
    )
    alternatives = EncodedBatch(
        torch.tensor([[0.6], [0.6]], dtype=torch.float64),
        torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.int64),
    )

    semantics = space.conditional_semantics
    assert semantics is not None
    activity = semantics.activity(alternatives).group
    assert torch.equal(activity[0], activity[1])
    mean, variance, _ = surrogate.predict(alternatives.continuous, alternatives.categorical)
    torch.testing.assert_close(mean[0], mean[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(variance[0], variance[1], rtol=0.0, atol=0.0)


def test_conditional_fantasy_update_appends_internal_activity() -> None:
    space = _branch_space()
    surrogate = _surrogate(space, fantasy=True)
    initial = EncodedBatch(
        torch.tensor([[0.0, 0.2], [0.5, 0.8], [1.0, 0.4]], dtype=torch.float64),
        torch.tensor([[1], [1], [0]], dtype=torch.int64),
    )
    targets = torch.tensor([0.0, 0.2, 0.8], dtype=torch.float64)
    surrogate.fit(initial.continuous, initial.categorical, targets, transform_version=0)
    surrogate.predict(initial.continuous, initial.categorical)

    appended = EncodedBatch(
        torch.cat(
            (
                initial.continuous,
                torch.tensor([[0.75, 0.95]], dtype=torch.float64),
            )
        ),
        torch.cat((initial.categorical, torch.tensor([[0]], dtype=torch.int64))),
    )
    report = surrogate.fit(
        appended.continuous,
        appended.categorical,
        torch.cat((targets, torch.tensor([0.7], dtype=torch.float64))),
        transform_version=0,
    )

    assert report.kind == "fantasy_update"
    assert surrogate.optimizer is not None
    assert not surrogate.optimizer.state
    assert surrogate.train_activity is not None
    assert surrogate.train_activity[:, 0].tolist() == [True, True, False, False]

# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

from __future__ import annotations

import pytest
import torch

import leanhebo.search.nsga2 as nsga2_module
from leanhebo.search.conditional import (
    ConditionalTorchNSGA2,
    eliminate_semantic_duplicates,
    semantic_duplicate_mask,
)
from leanhebo.search.conditional_operators import conditional_mutation
from leanhebo.search.operators import mixed_variable_crossover
from leanhebo.search.repair import MixedVariableSpec


class _BranchSemantics:
    def __init__(self, spec: MixedVariableSpec) -> None:
        self.spec = spec

    def activity_mask(self, population: torch.Tensor) -> torch.Tensor:
        active = torch.ones_like(population, dtype=torch.bool)
        active[:, 1] = population[:, 0] == 1
        return active

    def semantic_keys(self, population: torch.Tensor) -> torch.Tensor:
        active = self.activity_mask(population)
        projected = torch.where(active, population, torch.zeros_like(population))
        return torch.cat((active.to(population.dtype), projected), dim=1)

    def finite_completion(self, count: int, *, existing: torch.Tensor) -> torch.Tensor:
        selector_values = (
            [int(self.spec.fixed_values[0].item())] if bool(self.spec.fixed_mask[0]) else [0, 1]
        )
        rows: list[list[float]] = []
        for selector in selector_values:
            child_values = [0, 1] if selector == 1 else [0]
            rows.extend([float(selector), float(child)] for child in child_values)
        candidates = torch.tensor(
            rows,
            device=existing.device,
            dtype=existing.dtype,
        )
        unseen = eliminate_semantic_duplicates(candidates, self, existing=existing)
        return unseen[:count]


def _branch_spec(
    *,
    device: torch.device | str = "cpu",
    fixed_selector: int | None = None,
) -> MixedVariableSpec:
    fixed_mask = torch.tensor(
        [fixed_selector is not None, False],
        device=device,
    )
    fixed_values = torch.tensor(
        [0 if fixed_selector is None else fixed_selector, 0],
        dtype=torch.float32,
        device=device,
    )
    return MixedVariableSpec(
        torch.zeros(2, device=device),
        torch.ones(2, device=device),
        categorical_mask=torch.ones(2, dtype=torch.bool, device=device),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )


def test_semantic_duplicates_ignore_inactive_latent_values_exactly() -> None:
    spec = MixedVariableSpec(
        torch.zeros(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        categorical_mask=torch.tensor([True, False]),
    )
    semantics = _BranchSemantics(spec)
    population = torch.tensor(
        [[0.0, 0.1], [0.0, 0.9], [1.0, 0.1], [1.0, 0.1000000001]],
        dtype=torch.float64,
    )

    mask = semantic_duplicate_mask(population, semantics)

    assert mask.tolist() == [False, True, False, False]
    assert torch.equal(
        eliminate_semantic_duplicates(population, semantics),
        population[[0, 2, 3]],
    )


def test_semantic_duplicates_honor_existing_rows() -> None:
    spec = _branch_spec()
    semantics = _BranchSemantics(spec)
    population = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    existing = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    assert semantic_duplicate_mask(population, semantics, existing=existing).tolist() == [
        True,
        False,
        True,
    ]


def test_latent_dense_crossover_crosses_inactive_genes() -> None:
    spec = _branch_spec()
    first = torch.tensor([[0.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])

    child_a, child_b = mixed_variable_crossover(
        first,
        second,
        spec,
        probability=1.0,
        dimension_probability=1.0,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(child_a, torch.tensor([[0.0, 1.0]]))
    assert torch.equal(child_b, torch.tensor([[0.0, 0.0]]))


def test_conditional_mutation_uses_active_nonfixed_count_per_row() -> None:
    spec = MixedVariableSpec(
        lower=torch.zeros(4),
        upper=torch.tensor([1.0, 2.0, 2.0, 2.0]),
        categorical_mask=torch.ones(4, dtype=torch.bool),
        fixed_mask=torch.tensor([False, False, False, True]),
        fixed_values=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    class Semantics:
        def activity_mask(self, population: torch.Tensor) -> torch.Tensor:
            active = torch.ones_like(population, dtype=torch.bool)
            active[:, 1] = population[:, 0] == 1
            return active

        def semantic_keys(self, population: torch.Tensor) -> torch.Tensor:
            return population

        def finite_completion(self, count: int, *, existing: torch.Tensor) -> torch.Tensor:
            del count
            return existing.new_empty((0, existing.shape[1]))

    population = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    mutable = torch.tensor(
        [[True, False, True, False], [True, True, True, False]],
    )
    probability = torch.tensor([[0.5], [1.0 / 3.0]])
    expected_generator = torch.Generator().manual_seed(23)
    expected_gate = (
        torch.rand(population.shape, generator=expected_generator) < probability
    ) & mutable

    mutated = conditional_mutation(
        population,
        spec,
        Semantics(),
        generator=torch.Generator().manual_seed(23),
    )

    assert torch.equal(mutated != population, expected_gate)
    assert torch.equal(mutated[:, 3], torch.ones(2))


def test_contextual_finite_completion_returns_every_available_semantic_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _branch_spec(fixed_selector=1)
    semantics = _BranchSemantics(spec)
    search = ConditionalTorchNSGA2(
        spec,
        semantics,
        population_size=3,
        generations=0,
    )
    sampler = nsga2_module._SobolPopulationSampler(spec, None)
    repeated = torch.tensor([[1.0, 0.0]])
    monkeypatch.setattr(
        sampler,
        "draw",
        lambda count: repeated.expand(count, -1).clone(),
    )

    population = search._initialize(
        sampler,
        spec,
        incumbents=repeated,
        initial_population=None,
    )

    assert population.shape == (2, 2)
    assert torch.equal(population[:, 0], torch.ones(2))
    assert torch.equal(torch.sort(population[:, 1]).values, torch.tensor([0.0, 1.0]))


def test_conditional_nsga_stops_at_semantic_space_exhaustion() -> None:
    spec = _branch_spec()
    semantics = _BranchSemantics(spec)
    result = ConditionalTorchNSGA2(
        spec,
        semantics,
        population_size=8,
        generations=3,
    ).minimize(
        lambda population: semantics.semantic_keys(population).sum(dim=1),
        generator=torch.Generator().manual_seed(31),
    )

    assert result.population.shape == (3, 2)
    assert not bool(semantic_duplicate_mask(result.population, semantics).any())
    assert result.generations == 0
    assert result.objective_calls == 1


def test_conditional_nsga_completes_generations_with_a_continuous_child() -> None:
    spec = MixedVariableSpec(
        torch.zeros(2),
        torch.ones(2),
        categorical_mask=torch.tensor([True, False]),
    )
    semantics = _BranchSemantics(spec)

    result = ConditionalTorchNSGA2(
        spec,
        semantics,
        population_size=12,
        generations=3,
    ).minimize(
        lambda population: torch.stack(
            (
                (population[:, 0] - 1).square() + population[:, 1].square(),
                (population[:, 1] - 0.7).square(),
            ),
            dim=1,
        ),
        generator=torch.Generator().manual_seed(47),
    )

    assert result.generations == 3
    assert result.objective_calls == 4
    assert result.population.shape == (12, 2)
    assert not bool(semantic_duplicate_mask(result.population, semantics).any())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_conditional_keys_and_search_stay_on_cuda() -> None:
    device = torch.device("cuda")
    spec = _branch_spec(device=device)
    semantics = _BranchSemantics(spec)
    population = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        device=device,
    )

    mask = semantic_duplicate_mask(population, semantics)
    result = ConditionalTorchNSGA2(
        spec,
        semantics,
        population_size=3,
        generations=1,
    ).minimize(
        lambda values: semantics.semantic_keys(values).sum(dim=1),
        generator=torch.Generator(device=device).manual_seed(11),
    )

    assert mask.device.type == "cuda"
    assert result.population.device.type == "cuda"

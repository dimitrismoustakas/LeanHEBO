# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

from __future__ import annotations

import math

import pytest
import torch

import leanhebo.search.nsga2 as nsga2_module
from leanhebo.config import LeanHEBOConfig, RuntimeConfig
from leanhebo.optimizer import LeanHEBO, _CompiledConditionalSearchSemantics
from leanhebo.search.conditional import (
    ConditionalTorchNSGA2,
    eliminate_semantic_duplicates,
    semantic_duplicate_mask,
)
from leanhebo.search.conditional_operators import conditional_mutation, semantic_lift
from leanhebo.search.repair import MixedVariableSpec, repair_population
from leanhebo.space import Bool, Categorical, Eq, Float, GreaterThan, Integer, Space


class _BranchSemantics:
    def __init__(self, spec: MixedVariableSpec) -> None:
        self.spec = spec

    def activity_mask(self, population: torch.Tensor) -> torch.Tensor:
        active = torch.ones_like(population, dtype=torch.bool)
        active[:, 1] = population[:, 0] == 1
        return active

    def canonicalize_dense(self, population: torch.Tensor) -> torch.Tensor:
        repaired = repair_population(population, self.spec)
        return torch.where(repaired == 0, torch.zeros_like(repaired), repaired)

    def project(self, population: torch.Tensor) -> torch.Tensor:
        canonical = self.canonicalize_dense(population)
        projected = torch.where(self.activity_mask(canonical), canonical, self.spec.lower)
        if self.spec.has_fixed:
            projected[:, self.spec.fixed_mask] = self.spec.fixed_values[self.spec.fixed_mask]
        return projected

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
    continuous_child: bool = False,
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
        categorical_mask=torch.tensor(
            [True, not continuous_child],
            dtype=torch.bool,
            device=device,
        ),
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


def test_semantic_lift_copies_a_donor_and_uses_shared_no_donor_completion() -> None:
    semantics = _BranchSemantics(_branch_spec(continuous_child=True))
    parent_a = torch.tensor([[1.0, 0.75], [0.0, 0.1]])
    parent_b = torch.tensor([[0.0, 0.2], [0.0, 0.9]])
    completion = torch.tensor([[0.0, 0.4], [0.0, 0.6]])

    lifted_a, lifted_b = semantic_lift(
        parent_a,
        parent_b,
        completion,
        semantics,
    )

    assert torch.equal(lifted_a[:, 1], torch.tensor([0.75, 0.6]))
    assert torch.equal(lifted_b[:, 1], torch.tensor([0.75, 0.6]))
    assert torch.equal(lifted_a[:, 0], parent_a[:, 0])
    assert torch.equal(lifted_b[:, 0], parent_b[:, 0])


def test_equivalent_inactive_seed_values_produce_identical_searches() -> None:
    spec = _branch_spec(continuous_child=True)
    semantics = _BranchSemantics(spec)
    first_seed = torch.tensor([[0.0, 0.1], [1.0, 0.25], [1.0, 0.75]])
    second_seed = first_seed.clone()
    second_seed[0, 1] = 0.9

    def run(seed: torch.Tensor) -> nsga2_module.NSGA2Result:
        return ConditionalTorchNSGA2(
            spec,
            semantics,
            population_size=3,
            generations=3,
        ).minimize(
            lambda population: torch.stack(
                (
                    population.sum(dim=1),
                    (population[:, 0] - 1).square() + (population[:, 1] - 0.5).square(),
                ),
                dim=1,
            ),
            initial_population=seed,
            generator=torch.Generator().manual_seed(29),
        )

    first = run(first_seed)
    second = run(second_seed)

    assert torch.equal(first.population, second.population)
    assert torch.equal(first.objectives, second.objectives)
    assert torch.equal(first.ranks, second.ranks)
    assert torch.equal(first.crowding, second.crowding)


def test_no_donor_completion_survives_selector_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _branch_spec(continuous_child=True)
    semantics = _BranchSemantics(spec)
    search = ConditionalTorchNSGA2(
        spec,
        semantics,
        population_size=1,
        crossover_probability=0.0,
        mutation_probability=1.0,
        eliminate_duplicate_points=False,
    )
    sampler = nsga2_module._SobolPopulationSampler(spec, None)
    monkeypatch.setattr(
        sampler,
        "draw",
        lambda count: torch.tensor([[0.0, 0.6]]).expand(count, -1).clone(),
    )

    offspring = search._make_offspring(
        torch.tensor([[0.0, 0.0]]),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1),
        spec,
        sampler,
        torch.Generator().manual_seed(41),
    )

    assert torch.equal(offspring, torch.tensor([[1.0, 0.6]]))


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

        def canonicalize_dense(self, population: torch.Tensor) -> torch.Tensor:
            return population

        def project(self, population: torch.Tensor) -> torch.Tensor:
            return population

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


def test_compiled_projection_canonicalizes_log_integer_aliases() -> None:
    optimizer = LeanHEBO(
        Space(
            Integer("parent", 1, 100, log=True),
            Bool("child", active_when=Eq("parent", 4)),
        ),
        config=LeanHEBOConfig(runtime=RuntimeConfig(dtype="float64")),
    )
    semantics = _CompiledConditionalSearchSemantics(
        optimizer.space,
        None,
        device=optimizer.device,
    )
    spec = optimizer._search_spec(None)
    other = math.log(8.0) / math.log(100.0)

    def run(alias: float) -> nsga2_module.NSGA2Result:
        initial = torch.tensor(
            [[alias, 1.0], [other, 0.0]],
            dtype=optimizer.dtype,
        )
        return ConditionalTorchNSGA2(
            spec,
            semantics,
            population_size=2,
            generations=2,
            eliminate_duplicate_points=False,
        ).minimize(
            lambda population: (
                semantics.semantic_keys(population).to(dtype=population.dtype).sum(dim=1)
            ),
            initial_population=initial,
            generator=torch.Generator().manual_seed(53),
        )

    first = run(math.log(4.1) / math.log(100.0))
    second = run(math.log(4.4) / math.log(100.0))

    assert torch.equal(first.population, second.population)
    assert torch.equal(first.objectives, second.objectives)


def test_compiled_projection_normalizes_signed_zero_and_is_idempotent() -> None:
    optimizer = LeanHEBO(
        Space(
            Float("parent", -1.0, 1.0),
            Bool("child", active_when=GreaterThan("parent", 0.5)),
        ),
        config=LeanHEBOConfig(runtime=RuntimeConfig(dtype="float64")),
    )
    semantics = _CompiledConditionalSearchSemantics(
        optimizer.space,
        None,
        device=optimizer.device,
    )

    projected = semantics.project(torch.tensor([[-0.0, 1.0]], dtype=optimizer.dtype))

    assert not bool(torch.signbit(projected[0, 0]))
    assert torch.equal(projected, torch.tensor([[0.0, 0.0]], dtype=optimizer.dtype))
    assert torch.equal(semantics.project(projected), projected)


def test_contextual_projection_retains_an_inactive_fixed_value() -> None:
    optimizer = LeanHEBO(
        Space(
            Categorical("branch", ("off", "on")),
            Float("child", 0.0, 1.0, active_when=Eq("branch", "on")),
        )
    )
    fixed = optimizer.space.compile_fixed({"child": 0.375})
    semantics = _CompiledConditionalSearchSemantics(
        optimizer.space,
        fixed,
        device=optimizer.device,
    )
    population = torch.tensor([[0.9, 0.0], [0.1, 1.0]], dtype=optimizer.dtype)

    projected = semantics.project(population)

    assert torch.equal(
        projected,
        torch.tensor([[0.375, 0.0], [0.375, 1.0]], dtype=optimizer.dtype),
    )
    assert semantics.activity_mask(projected).tolist() == [[False, True], [True, True]]
    assert torch.equal(semantics.project(projected), projected)


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
    repeated = torch.tensor([[1.0, 0.0]])
    monkeypatch.setattr(
        nsga2_module._SobolPopulationSampler,
        "draw",
        lambda _sampler, count: repeated.expand(count, -1).clone(),
    )

    result = search.minimize(
        lambda population: population.sum(dim=1),
        incumbents=repeated,
    )
    population = result.population

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

    def objective(population: torch.Tensor) -> torch.Tensor:
        assert torch.equal(population, semantics.project(population))
        return torch.stack(
            (
                (population[:, 0] - 1).square() + population[:, 1].square(),
                (population[:, 1] - 0.7).square(),
            ),
            dim=1,
        )

    result = ConditionalTorchNSGA2(
        spec,
        semantics,
        population_size=12,
        generations=3,
    ).minimize(
        objective,
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

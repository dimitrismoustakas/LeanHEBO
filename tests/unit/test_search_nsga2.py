# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

import pytest
import torch

import leanhebo.search.nsga2 as nsga2_module
from leanhebo.search import (
    MixedVariableSpec,
    TorchNSGA2,
    crowding_distance,
    duplicate_mask,
    repair_population,
    sobol_population,
)
from leanhebo.space import Bool, Categorical, Float, Integer, Space


def _biobjective(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            (values[:, 0] - 0.2).square() + values[:, 1].square(),
            (values[:, 0] - 0.8).square() + (values[:, 1] - 1.0).square(),
        ),
        dim=1,
    )


def test_minimize_is_reproducible_from_a_torch_generator() -> None:
    optimizer = TorchNSGA2(population_size=24, generations=8)
    lower = torch.zeros(2)
    upper = torch.ones(2)

    first = optimizer.minimize(
        _biobjective,
        lower,
        upper,
        generator=torch.Generator().manual_seed(1234),
    )
    second = optimizer.minimize(
        _biobjective,
        lower,
        upper,
        generator=torch.Generator().manual_seed(1234),
    )

    assert torch.equal(first.population, second.population)
    assert torch.equal(first.objectives, second.objectives)
    assert torch.equal(first.ranks, second.ranks)
    assert first.population.shape == (24, 2)
    assert first.objectives.shape == (24, 2)
    torch.testing.assert_close(
        first.crowding,
        crowding_distance(first.objectives, first.ranks),
    )
    assert first.generations == 8
    assert first.objective_calls == 9
    assert first.candidate_evaluations == 24 * 9
    assert first.pareto_population.shape[0] > 1


def test_compiled_space_metadata_adapter_preserves_coordinate_semantics() -> None:
    compiled = Space(
        Float("real", 0.0, 1.0),
        Integer("stepped", 4, 12, step=4),
        Integer("power", 1, 100, log=True),
        Integer("exponent", 1, 8, base=2, exponent=True),
        Categorical("category", ("a", "b", "c")),
        Bool("flag"),
    ).compile()
    fixed = compiled.compile_fixed({"stepped": 8, "category": "b"})
    spec = MixedVariableSpec.from_compiled_space(
        compiled,
        fixed_mask=compiled.fixed_mask(fixed),
        fixed_values=compiled.dense_fixed_values(fixed),
    )

    population = sobol_population(
        spec,
        16,
        generator=torch.Generator().manual_seed(31),
    )

    assert spec.integer_mask.tolist() == [False, True, False, True, False, False]
    assert torch.equal(population[:, 1], torch.ones(16))  # coordinate 1 decodes to value 8
    assert torch.equal(population[:, 4], torch.ones(16))  # category code for "b"
    assert bool((population[:, 2] != population[:, 2].round()).any())  # power stays continuous


def test_minimize_injects_incumbents_before_sobol_points() -> None:
    optimizer = TorchNSGA2(
        population_size=8,
        generations=0,
        eliminate_duplicates=True,
    )
    incumbent = torch.tensor([0.25, 0.75])

    result = optimizer.minimize(
        _biobjective,
        torch.zeros(2),
        torch.ones(2),
        incumbent=incumbent,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(result.population[0], incumbent)
    assert result.generations == 0
    assert result.objective_calls == 1
    assert result.candidate_evaluations == 8


def test_mixed_variable_minimize_preserves_steps_categories_and_context() -> None:
    spec = MixedVariableSpec(
        lower=torch.tensor([0.0, 0.0, 0.0, -1.0]),
        upper=torch.tensor([1.0, 6.0, 2.0, 1.0]),
        integer_mask=torch.tensor([False, True, False, False]),
        categorical_mask=torch.tensor([False, False, True, False]),
        steps=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        fixed_mask=torch.tensor([False, False, False, True]),
        fixed_values=torch.tensor([0.0, 0.0, 0.0, 0.25]),
    )

    def objective(population: torch.Tensor) -> torch.Tensor:
        assert torch.equal(population, repair_population(population, spec))
        return torch.stack(
            (
                population[:, 0].square() + population[:, 1] + population[:, 2],
                (population[:, 0] - 1).square()
                + (population[:, 1] - 6).abs()
                + (population[:, 2] - 2).abs(),
            ),
            dim=1,
        )

    result = TorchNSGA2(spec, population_size=20, generations=5).minimize(
        objective,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(result.population[:, 1] % 2.0, torch.zeros(20))
    assert torch.equal(result.population[:, 2], torch.round(result.population[:, 2]))
    assert torch.equal(result.population[:, 3], torch.full((20,), 0.25))
    assert bool(((result.population >= spec.lower) & (result.population <= spec.upper)).all())


def test_combined_parent_offspring_pool_is_unique_before_survival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_survival = nsga2_module.elitist_survival
    checked_generations = 0

    def checked_survival(
        population: torch.Tensor,
        objectives: torch.Tensor,
        n_survive: int,
        **options: object,
    ) -> object:
        nonlocal checked_generations
        assert not bool(duplicate_mask(population).any())
        checked_generations += 1
        return original_survival(population, objectives, n_survive, **options)

    monkeypatch.setattr(nsga2_module, "elitist_survival", checked_survival)
    optimizer = TorchNSGA2(population_size=24, generations=6)

    optimizer.minimize(
        _biobjective,
        torch.zeros(2),
        torch.ones(2),
        generator=torch.Generator().manual_seed(317),
    )

    assert checked_generations == 6


def test_saturated_discrete_space_returns_all_available_unique_points() -> None:
    spec = MixedVariableSpec(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([1.0]),
        categorical_mask=torch.tensor([True]),
    )
    optimizer = TorchNSGA2(spec, population_size=10, generations=3, max_duplicate_retries=1)

    result = optimizer.minimize(
        lambda population: population.square(),
        generator=torch.Generator().manual_seed(8),
    )

    assert result.population.shape == (2, 1)
    assert torch.equal(torch.sort(result.population[:, 0]).values, torch.tensor([0.0, 1.0]))
    assert result.candidate_evaluations >= result.population.shape[0]


def test_objective_shape_and_finite_values_are_checked() -> None:
    optimizer = TorchNSGA2(population_size=4, generations=0)

    with pytest.raises(ValueError, match="shape"):
        optimizer.minimize(
            lambda population: torch.ones(3, 2),
            torch.zeros(1),
            torch.ones(1),
        )
    with pytest.raises(ValueError, match="non-finite"):
        optimizer.minimize(
            lambda population: torch.full((population.shape[0], 1), torch.nan),
            torch.zeros(1),
            torch.ones(1),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_minimize_uses_cuda_generator_and_tensors() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(29)
    result = TorchNSGA2(population_size=16, generations=2).minimize(
        _biobjective,
        torch.zeros(2, device=device),
        torch.ones(2, device=device),
        generator=generator,
    )

    assert result.population.device.type == "cuda"
    assert result.objectives.device.type == "cuda"

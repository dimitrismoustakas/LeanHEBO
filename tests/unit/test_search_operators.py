# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

import torch

from leanhebo.search import (
    MixedVariableSpec,
    binary_tournament,
    categorical_mutation,
    duplicate_mask,
    mixed_variable_crossover,
    mutate_population,
    repair_population,
    sbx_crossover,
)


def _mixed_spec() -> MixedVariableSpec:
    return MixedVariableSpec(
        lower=torch.tensor([0.0, 0.0, 0.0, 0.0, -2.0]),
        upper=torch.tensor([1.0, 6.0, 2.0, 1.0, 2.0]),
        integer_mask=torch.tensor([False, True, False, False, True]),
        categorical_mask=torch.tensor([False, False, True, True, False]),
        steps=torch.tensor([0.0, 2.0, 0.0, 0.0, 1.0]),
        fixed_mask=torch.tensor([False, False, False, False, True]),
        fixed_values=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_repair_is_idempotent_and_canonicalizes_every_variable_kind() -> None:
    spec = _mixed_spec()
    population = torch.tensor(
        [
            [-1.0, 1.1, 2.8, -0.4, -2.0],
            [2.0, 5.4, -0.2, 1.8, 2.0],
        ]
    )

    repaired = repair_population(population, spec)

    assert torch.equal(
        repaired,
        torch.tensor([[0.0, 2.0, 2.0, 0.0, 1.0], [1.0, 6.0, 0.0, 1.0, 1.0]]),
    )
    assert torch.equal(repair_population(repaired, spec), repaired)


def test_step_repair_never_clips_to_an_off_lattice_upper_bound() -> None:
    spec = MixedVariableSpec(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([5.0]),
        integer_mask=torch.tensor([True]),
        steps=torch.tensor([2.0]),
    )

    repaired = repair_population(torch.tensor([[4.9], [5.0]]), spec)

    assert torch.equal(repaired, torch.tensor([[4.0], [4.0]]))


def test_duplicate_detection_uses_repaired_integer_codes_and_stable_order() -> None:
    spec = _mixed_spec()
    population = torch.tensor(
        [
            [0.25, 1.1, 1.0, 0.0, 1.0],
            [0.25, 1.2, 1.1, 0.1, 1.5],
            [0.75, 1.2, 1.0, 0.0, 1.0],
        ]
    )

    mask = duplicate_mask(population, spec=spec)

    assert mask.tolist() == [False, True, False]


def test_exact_duplicate_detection_preserves_existing_priority_and_signed_zero() -> None:
    population = torch.tensor(
        [
            [-0.0, 1.0],
            [0.0, 1.0],
            [2.0, 3.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=torch.float64,
    )
    existing = torch.tensor([[2.0, 3.0], [9.0, 9.0]], dtype=torch.float64)

    mask = duplicate_mask(population, existing=existing)

    assert mask.tolist() == [False, True, True, True, False]


def test_exact_duplicate_detection_supports_empty_dimensions() -> None:
    population = torch.empty((3, 0))

    assert duplicate_mask(population).tolist() == [False, True, True]
    assert duplicate_mask(population, existing=torch.empty((1, 0))).tolist() == [True] * 3


def test_exact_duplicate_detection_matches_pairwise_reference() -> None:
    generator = torch.Generator().manual_seed(113)
    population = torch.randint(-3, 4, (200, 20), generator=generator).to(torch.float32)
    existing = torch.randint(-3, 4, (100, 20), generator=generator).to(torch.float32)
    combined = torch.cat((existing, population), dim=0)
    equal = (population[:, None, :] == combined[None, :, :]).all(dim=-1)
    combined_indices = torch.arange(combined.shape[0])
    population_indices = combined_indices[existing.shape[0] :]
    expected = (equal & (combined_indices[None, :] < population_indices[:, None])).any(dim=1)

    assert torch.equal(duplicate_mask(population, existing=existing), expected)


def test_categorical_mutation_always_chooses_an_alternative_at_probability_one() -> None:
    population = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    lower = torch.tensor([0.0, 0.0])
    upper = torch.tensor([2.0, 2.0])
    mask = torch.tensor([True, True])

    mutated = categorical_mutation(
        population,
        lower,
        upper,
        mask,
        probability=1.0,
        generator=torch.Generator().manual_seed(9),
    )

    assert bool((mutated != population).all())
    assert bool(((mutated >= lower) & (mutated <= upper)).all())
    assert torch.equal(mutated, torch.round(mutated))


def test_sbx_and_mixed_mutation_are_deterministic_and_respect_repair() -> None:
    spec = _mixed_spec()
    parent_a = torch.tensor([[0.1, 0.0, 0.0, 0.0, 1.0]]).repeat(8, 1)
    parent_b = torch.tensor([[0.9, 6.0, 2.0, 1.0, 1.0]]).repeat(8, 1)

    def operate(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        first, second = mixed_variable_crossover(
            parent_a,
            parent_b,
            spec,
            probability=1.0,
            dimension_probability=1.0,
            generator=generator,
        )
        return mutate_population(
            torch.cat((first, second)),
            spec,
            probability=1.0,
            generator=generator,
        )

    result = operate(17)

    assert torch.equal(result, operate(17))
    assert bool(((result >= spec.lower) & (result <= spec.upper)).all())
    assert torch.equal(result[:, 1] % 2.0, torch.zeros(result.shape[0]))
    assert torch.equal(result[:, 2:4], torch.round(result[:, 2:4]))
    assert torch.equal(result[:, 4], torch.ones(result.shape[0]))


def test_sbx_leaves_unselected_dimensions_unchanged() -> None:
    first = torch.tensor([[0.1, 0.2]])
    second = torch.tensor([[0.9, 0.8]])
    child_a, child_b = sbx_crossover(
        first,
        second,
        torch.zeros(2),
        torch.ones(2),
        numeric_mask=torch.tensor([True, False]),
        probability=1.0,
        dimension_probability=1.0,
        generator=torch.Generator().manual_seed(4),
    )

    assert child_a[0, 1].item() == first[0, 1].item()
    assert child_b[0, 1].item() == second[0, 1].item()


def test_binary_tournament_replays_exactly_from_generator_state() -> None:
    ranks = torch.tensor([0, 1, 0, 2])
    crowding = torch.tensor([0.0, 100.0, 1.0, 1000.0])

    first = binary_tournament(
        ranks,
        crowding,
        50,
        generator=torch.Generator().manual_seed(42),
    )
    second = binary_tournament(
        ranks,
        crowding,
        50,
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(first, second)
    assert bool(((first >= 0) & (first < ranks.numel())).all())

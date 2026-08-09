# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from leanhebo.search import (
    crowding_distance,
    dominance_matrix,
    non_dominated_fronts,
    non_dominated_sort,
    select_survivors,
)


def test_non_dominated_sort_assigns_expected_fronts() -> None:
    objectives = torch.tensor(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )

    dominates = dominance_matrix(objectives)
    ranks = non_dominated_sort(objectives)
    fronts = non_dominated_fronts(objectives)

    assert not bool(torch.diagonal(dominates).any())
    assert torch.equal(ranks, torch.tensor([0, 0, 0, 1, 2]))
    assert [front.tolist() for front in fronts] == [[0, 1, 2], [3], [4]]


def test_dominance_matrix_matches_strict_pairwise_reference_with_ties() -> None:
    generator = torch.Generator().manual_seed(83)
    for _ in range(20):
        # A small integer range deliberately creates duplicate rows and tied coordinates.
        objectives = torch.randint(-2, 3, (64, 4), generator=generator).to(torch.float32)
        no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(dim=-1)
        strictly_better = (objectives[:, None, :] < objectives[None, :, :]).any(dim=-1)
        expected = no_worse & strictly_better

        assert torch.equal(dominance_matrix(objectives), expected)


def test_crowding_distance_is_normalized_with_infinite_boundaries() -> None:
    objectives = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])

    distance = crowding_distance(objectives)

    assert torch.isinf(distance[[0, 2]]).all()
    assert distance[1].item() == 2.0


def test_crowding_ignores_constant_objective_ranges() -> None:
    objectives = torch.tensor([[0.0, 4.0], [1.0, 4.0], [2.0, 4.0]])

    distance = crowding_distance(objectives)

    assert torch.isinf(distance[[0, 2]]).all()
    assert distance[1].item() == 1.0


def test_ranked_crowding_matches_independent_front_calculation() -> None:
    generator = torch.Generator().manual_seed(19)
    objectives = torch.rand((80, 4), generator=generator)
    objectives[8:12, 2] = 0.5  # Exercise stable ties within a front.
    ranks = torch.tensor([0] * 17 + [1] * 2 + [2] + [4] * 25 + [7] * 35)

    expected = torch.empty(80)
    for rank in torch.unique(ranks, sorted=True):
        indices = torch.nonzero(ranks == rank, as_tuple=False).flatten()
        expected[indices] = crowding_distance(objectives[indices])

    actual = crowding_distance(objectives, ranks)

    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    assert torch.allclose(actual[torch.isfinite(expected)], expected[torch.isfinite(expected)])


def test_survival_takes_full_front_then_most_isolated_split_front() -> None:
    objectives = torch.tensor(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )

    selection = select_survivors(objectives, 4)

    assert selection.indices[:3].tolist() == [0, 1, 2]
    assert selection.indices[3].item() == 3


def test_empty_objective_matrix_has_empty_ranks_and_crowding() -> None:
    objectives = torch.empty((0, 3))

    assert non_dominated_sort(objectives).shape == (0,)
    assert crowding_distance(objectives).shape == (0,)


def test_non_dominated_ranks_match_pymoo_reference() -> None:
    generator = torch.Generator().manual_seed(37)
    objectives = torch.rand((40, 3), generator=generator)
    objectives[5] = objectives[4]

    pymoo_fronts = NonDominatedSorting().do(objectives.numpy())
    expected = torch.empty(40, dtype=torch.long)
    for rank, front in enumerate(pymoo_fronts):
        expected[torch.as_tensor(front, dtype=torch.long)] = rank

    assert torch.equal(non_dominated_sort(objectives), expected)

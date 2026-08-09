# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

import torch

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

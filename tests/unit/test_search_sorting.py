# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

import pytest
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from leanhebo.search.sorting import (
    _crowding_unchecked,
    _dominance_matrix_unchecked,
    _non_dominated_sort_unchecked,
)
from leanhebo.search.survival import _select_survivors_unchecked


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

    dominates = _dominance_matrix_unchecked(objectives)
    ranks, _ = _non_dominated_sort_unchecked(objectives)

    assert not bool(torch.diagonal(dominates).any())
    assert torch.equal(ranks, torch.tensor([0, 0, 0, 1, 2]))


def test_dominance_matrix_matches_strict_pairwise_reference_with_ties() -> None:
    generator = torch.Generator().manual_seed(83)
    for _ in range(20):
        # A small integer range deliberately creates duplicate rows and tied coordinates.
        objectives = torch.randint(-2, 3, (64, 4), generator=generator).to(torch.float32)
        no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(dim=-1)
        strictly_better = (objectives[:, None, :] < objectives[None, :, :]).any(dim=-1)
        expected = no_worse & strictly_better

        assert torch.equal(_dominance_matrix_unchecked(objectives), expected)


def test_crowding_distance_is_normalized_with_infinite_boundaries() -> None:
    objectives = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])

    distance = _crowding_unchecked(objectives)

    assert torch.isinf(distance[[0, 2]]).all()
    assert distance[1].item() == 2.0


def test_crowding_ignores_constant_objective_ranges() -> None:
    objectives = torch.tensor([[0.0, 4.0], [1.0, 4.0], [2.0, 4.0]])

    distance = _crowding_unchecked(objectives)

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
        expected[indices] = _crowding_unchecked(objectives[indices])

    actual = _crowding_unchecked(objectives, ranks)

    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    assert torch.allclose(actual[torch.isfinite(expected)], expected[torch.isfinite(expected)])


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
            ],
        ),
    ],
)
def test_crowding_preserves_objective_accumulation_order(device: str) -> None:
    generator = torch.Generator().manual_seed(7132)
    for n_objectives in (3, 12):
        values = torch.rand((113, n_objectives), generator=generator).to(device)
        groups = torch.randint(0, 3, (len(values),), generator=generator).to(device)
        for ranks in (None, groups):
            expected = values.new_zeros(len(values))
            for column in values.split(1, dim=1):
                expected += _crowding_unchecked(column, ranks)
            assert torch.equal(_crowding_unchecked(values, ranks), expected)


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

    selection = _select_survivors_unchecked(objectives, 4)

    assert selection.indices[:3].tolist() == [0, 1, 2]
    assert selection.indices[3].item() == 3


def test_truncated_survival_matches_full_ranking_on_every_retained_point() -> None:
    generator = torch.Generator().manual_seed(211)
    objectives = torch.rand((200, 3), generator=generator)
    full_ranks, _ = _non_dominated_sort_unchecked(objectives)
    full_crowding = _crowding_unchecked(objectives, full_ranks)
    n_survive = 100

    expected: list[torch.Tensor] = []
    selected_count = 0
    for rank in torch.unique(full_ranks, sorted=True):
        front = torch.nonzero(full_ranks == rank, as_tuple=False).flatten()
        remaining = n_survive - selected_count
        if front.numel() <= remaining:
            expected.append(front)
            selected_count += front.numel()
            continue
        order = torch.argsort(full_crowding[front], descending=True, stable=True)
        expected.append(front[order[:remaining]])
        break

    selection = _select_survivors_unchecked(objectives, n_survive)
    expected_indices = torch.cat(expected)

    assert torch.equal(selection.indices, expected_indices)
    assert torch.equal(selection.ranks[selection.indices], full_ranks[expected_indices])
    actual_crowding = selection.crowding[selection.indices]
    expected_crowding = full_crowding[expected_indices]
    assert torch.equal(torch.isinf(actual_crowding), torch.isinf(expected_crowding))
    assert torch.equal(
        actual_crowding[torch.isfinite(actual_crowding)],
        expected_crowding[torch.isfinite(expected_crowding)],
    )


@pytest.mark.parametrize("tied_coordinates", [False, True])
def test_non_dominated_ranks_match_pymoo_reference(tied_coordinates: bool) -> None:
    generator = torch.Generator().manual_seed(37)
    objectives = torch.rand((40, 3), generator=generator)
    if tied_coordinates:
        objectives = (objectives * 3).round()
    objectives[5] = objectives[4]

    pymoo_fronts = NonDominatedSorting().do(objectives.numpy())
    expected = torch.empty(40, dtype=torch.long)
    for rank, front in enumerate(pymoo_fronts):
        expected[torch.as_tensor(front, dtype=torch.long)] = rank

    ranks, _ = _non_dominated_sort_unchecked(objectives)
    assert torch.equal(ranks, expected)
    for count in range(1, len(expected) + 1):
        cutoff = int(expected.kthvalue(count).values)
        truncated, tail = _non_dominated_sort_unchecked(objectives, stop_after=count)
        assert tail == (cutoff + 1 if cutoff < expected.max() else None)
        assert torch.equal(truncated, expected.clamp_max(cutoff + 1))

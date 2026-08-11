# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Dense, tensor-native mixed-variable NSGA-II search."""

from leanhebo.search.duplicates import (
    duplicate_mask,
    eliminate_duplicates,
)
from leanhebo.search.nsga2 import NSGA2Result, TorchNSGA2
from leanhebo.search.operators import (
    binary_tournament,
    categorical_mutation,
    mixed_variable_crossover,
    mutate_population,
    polynomial_mutation,
    sbx_crossover,
    uniform_categorical_crossover,
)
from leanhebo.search.repair import MixedVariableSpec, repair_population
from leanhebo.search.sorting import (
    crowding_distance,
    dominance_matrix,
    non_dominated_sort,
)
from leanhebo.search.survival import (
    SurvivalResult,
    SurvivalSelection,
    elitist_survival,
    select_survivors,
)

__all__ = [
    "MixedVariableSpec",
    "NSGA2Result",
    "SurvivalResult",
    "SurvivalSelection",
    "TorchNSGA2",
    "binary_tournament",
    "categorical_mutation",
    "crowding_distance",
    "dominance_matrix",
    "duplicate_mask",
    "eliminate_duplicates",
    "elitist_survival",
    "mixed_variable_crossover",
    "mutate_population",
    "non_dominated_sort",
    "polynomial_mutation",
    "repair_population",
    "sbx_crossover",
    "select_survivors",
    "uniform_categorical_crossover",
]

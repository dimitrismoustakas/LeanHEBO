# SPDX-License-Identifier: MIT
# Portions derived from Huawei HEBO; see NOTICE.md.

"""Dense, tensor-native mixed-variable NSGA-II search."""

from leanhebo.search.duplicates import (
    canonical_duplicate_mask,
    duplicate_mask,
    eliminate_duplicates,
)
from leanhebo.search.nsga2 import (
    NSGA2Result,
    TorchNSGA2,
    initialize_sobol_population,
    sobol_population,
)
from leanhebo.search.operators import (
    binary_tournament,
    categorical_mutation,
    mixed_variable_crossover,
    mutate_population,
    numeric_mutation,
    polynomial_mutation,
    sbx_crossover,
    simulated_binary_crossover,
    uniform_categorical_crossover,
)
from leanhebo.search.repair import (
    CompiledSearchMetadata,
    MixedVariableSpec,
    repair,
    repair_population,
)
from leanhebo.search.sorting import (
    crowding_distance,
    dominance_matrix,
    fast_non_dominated_sort,
    non_dominated_fronts,
    non_dominated_ranks,
    non_dominated_sort,
)
from leanhebo.search.survival import (
    SurvivalResult,
    SurvivalSelection,
    elitist_survival,
    select_survivors,
    survival_indices,
)

__all__ = [
    "CompiledSearchMetadata",
    "MixedVariableSpec",
    "NSGA2Result",
    "SurvivalResult",
    "SurvivalSelection",
    "TorchNSGA2",
    "binary_tournament",
    "canonical_duplicate_mask",
    "categorical_mutation",
    "crowding_distance",
    "dominance_matrix",
    "duplicate_mask",
    "eliminate_duplicates",
    "elitist_survival",
    "fast_non_dominated_sort",
    "initialize_sobol_population",
    "mixed_variable_crossover",
    "mutate_population",
    "non_dominated_fronts",
    "non_dominated_ranks",
    "non_dominated_sort",
    "numeric_mutation",
    "polynomial_mutation",
    "repair",
    "repair_population",
    "sbx_crossover",
    "select_survivors",
    "simulated_binary_crossover",
    "sobol_population",
    "survival_indices",
    "uniform_categorical_crossover",
]

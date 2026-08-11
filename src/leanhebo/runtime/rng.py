# SPDX-License-Identifier: MIT

"""Independent random streams for sampling, acquisition, search, and selection."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import torch

_GENERATOR_SEED_MODULUS = 2**63 - 1
_SOBOL_SEED_MODULUS = 2**31 - 1


def make_generator(device: torch.device | str, seed: int | None) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


@dataclass(slots=True)
class RandomStreams:
    sobol_seed: int
    acquisition: torch.Generator
    model: torch.Generator
    search: torch.Generator
    selection: torch.Generator

    @classmethod
    def create(
        cls,
        device: torch.device | str,
        seed: int | None,
        search_seed: int | None = None,
    ) -> RandomStreams:
        # ``torch.seed()`` reseeds the process-global generator.  Entropy for an
        # unseeded optimizer must not perturb unrelated Torch callers.
        root = secrets.randbelow(_GENERATOR_SEED_MODULUS) if seed is None else seed
        return cls(
            sobol_seed=root % _SOBOL_SEED_MODULUS,
            acquisition=make_generator(device, (root + 1) % _GENERATOR_SEED_MODULUS),
            model=make_generator(device, (root + 2) % _GENERATOR_SEED_MODULUS),
            search=make_generator(
                device,
                (root + 3) % _GENERATOR_SEED_MODULUS if search_seed is None else search_seed,
            ),
            selection=make_generator(device, (root + 4) % _GENERATOR_SEED_MODULUS),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "sobol_seed": self.sobol_seed,
            "acquisition": self.acquisition.get_state(),
            "model": self.model.get_state(),
            "search": self.search.get_state(),
            "selection": self.selection.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        sobol_seed = state["sobol_seed"]
        if not isinstance(sobol_seed, int):
            raise TypeError("invalid Sobol seed state")
        acquisition = state["acquisition"]
        model = state["model"]
        search = state["search"]
        selection = state["selection"]
        if not all(
            isinstance(item, torch.Tensor) for item in (acquisition, model, search, selection)
        ):
            raise TypeError("invalid Torch generator state")
        self.sobol_seed = sobol_seed
        self.acquisition.set_state(acquisition)  # type: ignore[arg-type]
        self.model.set_state(model)  # type: ignore[arg-type]
        self.search.set_state(search)  # type: ignore[arg-type]
        self.selection.set_state(selection)  # type: ignore[arg-type]

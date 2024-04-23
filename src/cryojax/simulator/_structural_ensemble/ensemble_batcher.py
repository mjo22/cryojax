from abc import abstractmethod

import equinox as eqx

from .base_ensemble import AbstractStructuralEnsemble


class AbstractStructuralEnsembleBatcher(eqx.Module, strict=True):
    """A batching utility for structural ensembles."""

    @abstractmethod
    def get_batched_structural_ensemble(self) -> AbstractStructuralEnsemble:
        raise NotImplementedError

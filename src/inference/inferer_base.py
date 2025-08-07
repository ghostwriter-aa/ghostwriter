from abc import ABC, abstractmethod
from typing import SupportsFloat

from common import common_types as ct


class InfererBase(ABC):
    """
    Base class for performing inference on a trained model.
    Subclasses should implement the `infer` method.
    Subclasses should be instantiated using the `inferer_factory` method in `inferer_factory.py`.
    """

    def __init__(self, model: ct.Model):
        self.model = model

    def prepare_inference(self, personas: dict[ct.PersonaId, ct.PersonaInfo]) -> None:
        """
        Optional preparation method for subclasses to precompute some persona-specific inference data.
        For example, this can be used to compute embeddings for each persona, so that two-persona matching can then
        be performed efficiently.
        """

    @abstractmethod
    def infer(
        self, persona1_id: ct.PersonaId, persona1: ct.PersonaInfo, persona2_id: ct.PersonaId, persona2: ct.PersonaInfo
    ) -> SupportsFloat:
        """
        Perform inference on the provided persona.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the infer method.")

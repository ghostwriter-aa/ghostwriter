from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from common import common_types as ct
from contrastive_learning.embeddings import EMBEDDING_MODEL_CLASSES, AveragePostsStrategy
from contrastive_learning.text_sim_clr import TextSimCLR, compute_feature_representation
from inference import inferer_base


class EmbeddingInferer(inferer_base.InfererBase):
    def __init__(self, model: ct.Model) -> None:
        super().__init__(model)
        if model.model_type != ct.ModelType.EMBEDDING_MODEL:
            raise ValueError(f"EmbeddingInferer can only be used with EMBEDDING_MODEL, but got {model.model_type}")
        model_params: ct.EmbeddingModelParams = cast(ct.EmbeddingModelParams, model.model_params)

        embedder_cls = EMBEDDING_MODEL_CLASSES[model_params.embedding_model]
        self.embedder = embedder_cls()
        self.embedding_strategy = AveragePostsStrategy()

        self.text_simclr_model = TextSimCLR.load_from_checkpoint(model_params.embedding_model_checkpoint_path)
        self.text_simclr_model.eval()  # The model is only used for inference.

        # This is used to store the pre-computed embeddings for the personas.
        self.persona_embeddings_by_id: dict[ct.PersonaId, ct.PersonaEmbedding] = {}

    def prepare_inference(self, personas: dict[ct.PersonaId, ct.PersonaInfo]) -> None:
        persona_id_list = personas.keys()
        persona_info_list = [personas[persona_id] for persona_id in persona_id_list]
        persona_embeddings = self.embedding_strategy.compute_persona_embeddings(
            persona_info_list, self.embedder, persona_chunk_size=512, batch_size=128
        )
        self.persona_embeddings_by_id.update(
            {
                persona_id: persona_embedding
                for persona_id, persona_embedding in zip(persona_id_list, persona_embeddings)
            }
        )

    def load_precomputed_embeddings(self, persona_embeddings_by_id: dict[ct.PersonaId, ct.PersonaEmbedding]) -> None:
        self.persona_embeddings_by_id.update(persona_embeddings_by_id)

    def infer(
        self, persona1_id: ct.PersonaId, persona1: ct.PersonaInfo, persona2_id: ct.PersonaId, persona2: ct.PersonaInfo
    ) -> float:
        """Return the cosine similarity (in feature space) of two personas."""
        # Either use pre-computed embeddings or compute them on the fly.
        persona_feature_vectors: list[NDArray[np.floating[Any]]] = []

        for persona_id, persona in (
            (persona1_id, persona1),
            (persona2_id, persona2),
        ):
            if persona_id in self.persona_embeddings_by_id:
                persona_embedding = self.persona_embeddings_by_id[persona_id]
            else:
                persona_embedding = self.embedding_strategy.compute_single_persona_embedding(
                    persona, self.embedder, batch_size=128
                )

            # Convert the numpy embedding to a torch tensor of shape (1, D) for the projection.
            embedding_tensor = torch.from_numpy(persona_embedding.embedding).unsqueeze(0)
            projected = compute_feature_representation(self.text_simclr_model, embedding_tensor)[0].cpu().numpy()
            persona_feature_vectors.append(projected)

        # Compute cosine similarity between the two feature vectors.
        return float(
            np.dot(persona_feature_vectors[0], persona_feature_vectors[1])
            / (np.linalg.norm(persona_feature_vectors[0]) * np.linalg.norm(persona_feature_vectors[1]))
        )

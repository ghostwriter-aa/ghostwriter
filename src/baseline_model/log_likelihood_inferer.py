from typing import SupportsFloat, cast

import tqdm

from baseline_model import token_stats
from common import common_types as ct
from common import tokenization
from inference import inferer_base


class LogLikelihoodInferer(inferer_base.InfererBase):
    def __init__(self, model: ct.Model) -> None:
        super().__init__(model)
        if model.model_type != ct.ModelType.LOG_LIKELIHOOD_MODEL:
            raise ValueError(
                f"LogLikelihoodInferer can only be used with LOG_LIKELIHOOD_MODEL, but got {model.model_type}"
            )
        model_params = cast(ct.LogLikelihoodModelParams, model.model_params)
        self.model_tokens = [token.token_index for token in model_params.tokens]
        self.tokenizer = tokenization.get_tokenizer()
        self.persona_stats: dict[ct.PersonaId, token_stats.TokenStats] = {}

    def prepare_inference(self, personas: dict[ct.PersonaId, ct.PersonaInfo]) -> None:
        for persona_id, persona_info in tqdm.tqdm(personas.items(), desc="Computing token statistics"):
            self.persona_stats[persona_id] = token_stats.TokenStats.from_persona_info(
                persona_info, self.model_tokens, self.tokenizer
            )

    def infer(
        self, persona1_id: ct.PersonaId, persona1: ct.PersonaInfo, persona2_id: ct.PersonaId, persona2: ct.PersonaInfo
    ) -> SupportsFloat:
        return self.persona_stats[persona1_id].log_likelihood(self.persona_stats[persona2_id])

from baseline_model import log_likelihood_inferer
from common import common_types as ct
from contrastive_learning import embedding_inferrer
from inference import inferer_base


def inferer_factory(model: ct.Model) -> inferer_base.InfererBase:
    match model.model_type:
        case ct.ModelType.LOG_LIKELIHOOD_MODEL:
            return log_likelihood_inferer.LogLikelihoodInferer(model)
        case ct.ModelType.EMBEDDING_MODEL:
            return embedding_inferrer.EmbeddingInferer(model)
        case _:
            raise ValueError(f"Unknown model type: {model.model_type}")

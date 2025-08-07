from common import common_types as ct
from common.common_types import ModelType


def test_load_log_likelihood_model() -> None:
    saved_model = {
        "model_type": "LOG_LIKELIHOOD_MODEL",
        "model_params": {
            "tokens": [
                {"token_index": 802, "string": "\u2019s"},
                {"token_index": 1202, "string": " \n\n"},
                {"token_index": 364, "string": ".\n\n"},
            ]
        },
    }
    model = ct.Model.from_json(saved_model)
    assert model.model_type == ModelType.LOG_LIKELIHOOD_MODEL
    assert isinstance(model.model_params, ct.LogLikelihoodModelParams)
    saved_tokens = saved_model["model_params"]["tokens"]  # type: ignore
    assert len(model.model_params.tokens) == len(saved_tokens)
    for model_token, saved_model_token in zip(model.model_params.tokens, saved_tokens):
        assert model_token.token_index == saved_model_token["token_index"]
        assert model_token.string == saved_model_token["string"]

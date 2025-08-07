import tempfile
from pathlib import Path

import pandas as pd
import pytest

from common import common_types as ct
from inference import run_inference


def test_parquet_to_persona_info() -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as temp_file:
        temp_file_name = temp_file.name

        # Write test data to the Parquet file
        data = {
            "document_id": ["doc0", "doc1", "doc2"],
            "entity": ["entity1", "entity1", "entity1"],
            "persona": ["persona1", "persona1", "persona2"],
            "text": [
                "This is a comment for persona1.",
                "This is another comment for persona1.",
                "This one is for persona2.",
            ],
        }
        df = pd.DataFrame(data).astype(
            {"document_id": "string", "entity": "string", "persona": "string", "text": "string"}
        )
        df.to_parquet(temp_file_name, index=False)

        personas = run_inference.parquet_to_persona_info(temp_file_name)

    # Check the results
    persona1id = ct.PersonaId("entity1", "persona1")
    persona2id = ct.PersonaId("entity1", "persona2")
    assert personas.keys() == {persona1id, persona2id}
    p1comments = [comment.body for comment in personas[persona1id].comments]
    assert p1comments == [
        "This is a comment for persona1.",
        "This is another comment for persona1.",
    ]
    p2comments = [comment.body for comment in personas[persona2id].comments]
    assert p2comments == ["This one is for persona2."]


# Global constants used across inference tests.
E1P1_ID = ct.PersonaId("jekyll", "jekyll")
E1P2_ID = ct.PersonaId("jekyll", "hyde")
E2P1_ID = ct.PersonaId("robert", "robert")

PERSONA_INFOS: dict[ct.PersonaId, ct.PersonaInfo] = {
    E1P1_ID: ct.PersonaInfo(
        subreddit="",
        comments=[
            ct.CommentInfo(body="This persona doesn't like saying positive statements.", permalink=""),
            ct.CommentInfo(body="They just don't.", permalink=""),
        ],
    ),
    E1P2_ID: ct.PersonaInfo(
        subreddit="",
        comments=[
            ct.CommentInfo(body="It doesn't matter what you say, just don't say a positive sentence.", permalink=""),
        ],
    ),
    E2P1_ID: ct.PersonaInfo(
        subreddit="",
        comments=[
            ct.CommentInfo(body="What does this button do?", permalink=""),
            ct.CommentInfo(body="A statement that contains no positive or negative tokens.", permalink=""),
        ],
    ),
}

PAIRS_TO_INFER: list[tuple[ct.PersonaId, ct.PersonaId]] = [
    (E1P1_ID, E1P2_ID),
    (E1P1_ID, E2P1_ID),
]


@pytest.mark.parametrize(
    "model, higher_score_indicates_better_match",
    [
        pytest.param(
            ct.Model.from_json(
                {
                    "model_type": "LOG_LIKELIHOOD_MODEL",
                    "model_params": {
                        "tokens": [
                            {"token_index": 2226, "string": " does"},
                            {"token_index": 621, "string": " do"},
                            {"token_index": 4128, "string": " don't"},
                            {"token_index": 8740, "string": " doesn't"},
                        ]
                    },
                }
            ),
            True,  # Log-likelihood baseline expects higher score for first pair
            id="log-likelihood",
        ),
        pytest.param(
            ct.Model.from_json(
                {
                    "model_type": "EMBEDDING_MODEL",
                    "model_params": {
                        "embedding_model": "e5",
                        "embedding_strategy": "average_posts",
                        "embedding_model_checkpoint_path": str(Path(__file__).resolve().parent / "TextSimCLR.pt"),
                    },
                }
            ),
            False,  # Contrastive learning model expects lower score for first pair
            id="contrastive-learning",
        ),
    ],
)
def test_run_inference(
    model: ct.Model,
    higher_score_indicates_better_match: bool,
) -> None:
    """Ensure that models return sensible match scores."""
    match_scores_df = run_inference.run_inference(model, PERSONA_INFOS, PAIRS_TO_INFER)

    assert len(match_scores_df) == 2

    # Build (PersonaId, PersonaId) -> score mapping
    scores = {
        (ct.PersonaId(r.entity1, r.persona1), ct.PersonaId(r.entity2, r.persona2)): r.match_score
        for _, r in match_scores_df.iterrows()
    }

    assert set(scores.keys()) == set(PAIRS_TO_INFER)

    matching_pair = PAIRS_TO_INFER[0]
    cross_pair = (PAIRS_TO_INFER[0][0], PAIRS_TO_INFER[1][1])

    if higher_score_indicates_better_match:
        assert scores[matching_pair] > scores[cross_pair]
    else:
        assert scores[matching_pair] < scores[cross_pair]

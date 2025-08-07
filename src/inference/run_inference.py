import argparse
from typing import Sequence, SupportsFloat

import pandas as pd
import pyarrow.parquet as pq
import tqdm

from common import common_types as ct
from inference import inferer_factory


def parquet_to_persona_info(parquet_file: str) -> dict[ct.PersonaId, ct.PersonaInfo]:
    """
    Reads a Parquet file and converts it to a dictionary of PersonaInfo objects keyed by (entity, persona).
    """
    columns_to_read = ["entity", "persona", "text"]
    df = pq.read_pandas(parquet_file, columns=columns_to_read).to_pandas()
    for column in columns_to_read:
        if df[column].dtype != "string":
            raise ValueError(
                f"Column {column} in Parquet file must be of type 'string', but found type '{df[column].dtype}'"
            )

    # Group by 'entity' and 'persona' to create a dictionary of lists
    personas: dict[ct.PersonaId, ct.PersonaInfo] = {}
    grouped = df.groupby(["entity", "persona"])
    for (entity_id, persona_id), group_df in grouped:
        persona_info = ct.PersonaInfo(
            subreddit="",
            comments=[ct.CommentInfo(body=text, permalink="") for text in group_df["text"].tolist()],
        )
        # Note: Our codebase uses the term `author` whereas the Parquet file uses `entity`.
        personas[ct.PersonaId(author=entity_id, persona=persona_id)] = persona_info

    return personas


def parquet_to_pairs_to_infer(parquet_file: str) -> list[tuple[ct.PersonaId, ct.PersonaId]]:
    """
    Reads a Parquet file and converts it to a list of pairs of PersonaId objects.
    Each pair represents two personas to be evaluated against each other.
    """
    columns_to_read = ["entity1", "persona1", "entity2", "persona2"]
    df = pq.read_pandas(parquet_file, columns=columns_to_read).to_pandas()
    for column in columns_to_read:
        if df[column].dtype != "string":
            raise ValueError(
                f"Column {column} in Parquet file must be of type 'string', but found type '{df[column].dtype}'"
            )

    pairs: list[tuple[ct.PersonaId, ct.PersonaId]] = []
    for _, row in df.iterrows():
        persona1 = ct.PersonaId(author=row["entity1"], persona=row["persona1"])
        persona2 = ct.PersonaId(author=row["entity2"], persona=row["persona2"])
        pairs.append((persona1, persona2))

    return pairs


def run_inference(
    model: ct.Model,
    persona_infos: dict[ct.PersonaId, ct.PersonaInfo],
    pairs_to_infer: Sequence[tuple[ct.PersonaId, ct.PersonaId]],
) -> pd.DataFrame:
    inferer = inferer_factory.inferer_factory(model)
    inferer.prepare_inference(persona_infos)
    inference_results: dict[str, list[str | SupportsFloat]] = {
        "entity1": [],
        "persona1": [],
        "entity2": [],
        "persona2": [],
        "match_score": [],
    }
    for persona1, persona2 in tqdm.tqdm(pairs_to_infer, desc="Running inference"):
        score = inferer.infer(persona1, persona_infos[persona1], persona2, persona_infos[persona2])
        inference_results["entity1"].append(persona1.author)
        inference_results["persona1"].append(persona1.persona)
        inference_results["entity2"].append(persona2.author)
        inference_results["persona2"].append(persona2.persona)
        inference_results["match_score"].append(score)

    return pd.DataFrame(inference_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Performs inference on a previously trained model.")
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="JSON file containing the trained model parameters.",
    )
    parser.add_argument(
        "--author_docs_file",
        type=str,
        required=True,
        help="Parquet file containing the documents for all authors to be evaluated.",
    )
    parser.add_argument(
        "--pairs_to_match_file",
        type=str,
        required=True,
        help="Parquet file containing the pairs of authors to be evaluated.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output Parquet file where the inference results will be written.",
    )
    args = parser.parse_args()

    with open(args.model_file, "rt") as f:
        model = ct.Model.from_json(f.read())
    persona_infos = parquet_to_persona_info(args.author_docs_file)
    pairs_to_infer = parquet_to_pairs_to_infer(args.pairs_to_match_file)

    df = run_inference(model, persona_infos, pairs_to_infer)
    df.to_parquet(args.output_file)


if __name__ == "__main__":
    main()

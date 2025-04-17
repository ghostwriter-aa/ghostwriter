"""Builds a dataset of documents by author containing all of their posts in each of two subreddits."""

import argparse
import collections
import dataclasses
import json

import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray

from common import common_types as ct
from data_handling import reddit_consts


def get_subreddit_adjacency_matrix(
    active_subreddits_file: str, filtered_submissions_file: str
) -> tuple[NDArray[np.int32], list[str], dict[str, dict[str, int]]]:
    """Builds an adjacency matrix between subreddits.

    Returns:
        adjacency_matrix: A square matrix where the (i, j) entry is the number of authors who posted in both
            subreddits i and j.
        active_subreddits_list: A list of subreddit names in the same order as the rows and columns of the adjacency
            matrix.
        subreddit_to_author_to_num_posts: A mapping from subreddit names to a mapping of authors and the number of
            submissions they have in that subreddit.
    """
    # Get very active subreddits: those with over 1000 submissions per month.
    active_subreddits_df = pd.read_csv(active_subreddits_file)
    active_subreddits_df = active_subreddits_df[active_subreddits_df["count"] > 1000]
    active_subreddits_list = active_subreddits_df["subreddit"].tolist()
    active_subreddits = set(active_subreddits_list)

    subreddit_to_author_to_num_posts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    num_legit_submissions = 0
    print("Finding list of authors by subreddit...")
    with open(filtered_submissions_file, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
            submission = json.loads(line)
            subreddit = submission["subreddit"]
            if subreddit not in active_subreddits:
                continue
            if submission["author"] in reddit_consts.USERS_TO_IGNORE:
                # We should be running on filtered data, so this is not expected to occur.
                # Keeping this here as long as we are still running on old versions of the data.
                continue
            num_legit_submissions += 1
            subreddit_to_author_to_num_posts[subreddit][submission["author"]] += 1

    print("Building adjacency matrix...")
    adjacency_matrix = np.zeros((len(active_subreddits_list), len(active_subreddits_list)), dtype=np.int32)
    for i, subreddit1 in enumerate(tqdm.tqdm(active_subreddits_list)):
        authors1 = set(subreddit_to_author_to_num_posts[subreddit1])
        for j, subreddit2 in enumerate(active_subreddits_list):
            if j >= i:
                break
            authors2 = set(subreddit_to_author_to_num_posts[subreddit2])
            adjacency_matrix[i, j] = len(authors1 & authors2)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T  # type: ignore
    return adjacency_matrix, active_subreddits_list, subreddit_to_author_to_num_posts


def get_suitable_authors(
    adjacency_matrix: NDArray[np.int32],
    active_subreddits_list: list[str],
    subreddit_to_author_to_num_posts: dict[str, dict[str, int]],
    min_author_posts_per_subreddit: int = 5,
) -> list[ct.AuthorInfo]:
    """Finds authors who *uniquely* post in two unrelated subreddits.

    Here, "uniquely" means no other author posts in those two subreddits.

    Args:
        adjacency_matrix, active_subreddits_list, subreddit_to_author_to_num_posts: As returned from
            get_subreddit_adjacency_matrix.
        min_author_posts_per_subreddit: Minimum number of submissions an author must have in two distinct channels in
            or to qualify as suitable.

    Returns:
        List of AuthorInfo objects with only the username and persona subreddits filled in
        (no submissions or comments).
    """
    author_to_author_info: dict[str, ct.AuthorInfo] = {}
    for subreddit1_idx, subreddit1 in enumerate(tqdm.tqdm(active_subreddits_list)):
        neighbors = adjacency_matrix[subreddit1_idx, :]
        if np.max(neighbors) < 20:
            # Not many neighbors for this subreddit, so skip it since we don't know for sure
            # which neighbors are truly distant.
            continue
        potential_matches = np.where(adjacency_matrix[subreddit1_idx, :] == 1)[0]
        for subreddit2_idx in potential_matches:
            subreddit2 = active_subreddits_list[subreddit2_idx]
            joint_authors = set(subreddit_to_author_to_num_posts[subreddit1]) & set(
                subreddit_to_author_to_num_posts[subreddit2]
            )
            assert len(joint_authors) == 1, f"Expected 1 author, got {joint_authors}"
            author = joint_authors.pop()
            if author in author_to_author_info:
                continue  # We already have two other personas for this author, so skip these.
            if subreddit_to_author_to_num_posts[subreddit1][author] < min_author_posts_per_subreddit:
                continue
            if subreddit_to_author_to_num_posts[subreddit2][author] < min_author_posts_per_subreddit:
                continue
            author_to_author_info[author] = ct.AuthorInfo(
                username=author,
                personas=[ct.PersonaInfo(subreddit=subreddit1), ct.PersonaInfo(subreddit=subreddit2)],
            )
    return list(author_to_author_info.values())


def get_submissions_for_authors(author_infos: list[ct.AuthorInfo], filtered_submissions_file: str) -> None:
    """Adds all author submissions for the relevant subreddits into the provided AuthorInfos."""
    author_info_by_username = {author_info.username: author_info for author_info in author_infos}
    print("Adding submissions to AuthorInfos...")
    with open(filtered_submissions_file, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
            submission = json.loads(line)
            author = submission["author"]
            author_info = author_info_by_username.get(author)
            if not author_info:
                continue
            subreddit = submission["subreddit"]
            author_subreddits = [persona.subreddit for persona in author_info.personas]
            if subreddit not in author_subreddits:
                continue
            persona_info = author_info.personas[author_subreddits.index(subreddit)]
            persona_info.submissions.append(
                ct.SubmissionInfo(title=submission["title"], selftext=submission["selftext"], url=submission["url"])
            )


def get_comments_for_authors(author_infos: list[ct.AuthorInfo], filtered_comments_file: str) -> None:
    """Adds all author comments for the relevant subreddits into the provided AuthorInfos."""
    author_info_by_username = {author_info.username: author_info for author_info in author_infos}
    print("Adding comments to AuthorInfos...")
    with open(filtered_comments_file, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RC_2024_05_FILTERED_COMMENTS):
            comment = json.loads(line)
            author = comment["author"]
            author_info = author_info_by_username.get(author)
            if not author_info:
                continue
            subreddit = comment["subreddit"]
            author_subreddits = [persona.subreddit for persona in author_info.personas]
            if subreddit not in author_subreddits:
                continue
            persona_info = author_info.personas[author_subreddits.index(subreddit)]
            persona_info.comments.append(ct.CommentInfo(body=comment["body"], permalink=comment["permalink"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output NDJSON file containing AuthorInfo objects."
    )
    parser.add_argument(
        "--input_submissions",
        type=str,
        default="../data/RS_2024-05_filtered",
        required=False,
        help="Path to the input filtered submissions file (e.g., ../data/RS_2024-05_filtered).",
    )
    parser.add_argument(
        "--input_comments",
        type=str,
        default="../data/RC_2024-05_filtered",
        required=False,
        help="Path to the input filtered comments file (e.g., ../data/RC_2024-05_filtered).",
    )
    parser.add_argument(
        "--input_active_subreddits",
        type=str,
        default="../summary_data/active_subreddits_cutoff_30.csv",
        required=False,
        help="Path to the input CSV file for active subreddits "
        "(e.g., ../summary_data/active_subreddits_cutoff_30.csv).",
    )
    args = parser.parse_args()

    adjacency_matrix, active_subreddits_list, subreddit_to_authors_to_num_posts = get_subreddit_adjacency_matrix(
        args.input_active_subreddits, args.input_submissions
    )
    suitable_authors = get_suitable_authors(adjacency_matrix, active_subreddits_list, subreddit_to_authors_to_num_posts)
    get_submissions_for_authors(suitable_authors, args.input_submissions)
    get_comments_for_authors(suitable_authors, args.input_comments)
    with open(args.output_file, "wt") as f:
        for author in suitable_authors:
            f.write(json.dumps(dataclasses.asdict(author)) + "\n")


if __name__ == "__main__":
    main()

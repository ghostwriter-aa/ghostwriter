"""Collect summary statistics for Reddit data: Most prolific authors and active subreddits."""

import argparse
import collections
import csv
import json

import tqdm

from data_handling import reddit_consts


def get_author_and_subreddit_counts(
    input_submissions_file: str, input_comments_file: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Returns the counts of author submissions/comments and subreddit submissions/comments in the filtered dataset."""
    # This function takes about 25 minutes to run (on an ec2 g5.xlarge).
    authors: dict[str, int] = collections.Counter()
    subreddits: dict[str, int] = collections.Counter()
    with open(input_submissions_file, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
            submission = json.loads(line)
            authors[submission["author"]] += 1
            subreddits[submission["subreddit"]] += 1
    with open(input_comments_file, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RC_2024_05_FILTERED_COMMENTS):
            comment = json.loads(line)
            authors[comment["author"]] += 1
            subreddits[comment["subreddit"]] += 1
    return authors, subreddits


def get_entries_with_min_submissions(entries: dict[str, int], min_submissions: int) -> dict[str, int]:
    """Returns entries with more than `min_submissions` submissions."""
    return {entry: count for entry, count in entries.items() if count >= min_submissions}


def write_csv(file_path: str, data: dict[str, int], column_titles: list[str]) -> None:
    """Writes 2-column data to a CSV file."""
    with open(file_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(column_titles)
        for key, value in data.items():
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect summary statistics for Reddit data.")
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
        "--output_authors",
        type=str,
        required=True,
        help="Path to the output CSV file for prolific authors (e.g., ../summary_data/prolific_authors_cutoff_30.csv).",
    )
    parser.add_argument(
        "--output_subreddits",
        type=str,
        required=True,
        help="Path to the output CSV file for active subreddits "
        "(e.g., ../summary_data/active_subreddits_cutoff_30.csv).",
    )
    parser.add_argument(
        "--author_cutoff",
        type=int,
        default=30,
        help="Minimum number of submissions for an author to be considered prolific.",
    )
    parser.add_argument(
        "--subreddit_cutoff",
        type=int,
        default=30,
        help="Minimum number of submissions for a subreddit to be considered active.",
    )
    args = parser.parse_args()

    authors, subreddits = get_author_and_subreddit_counts(args.input_submissions, args.input_comments)
    print(f"There are {len(authors)} authors in the filtered dataset.")
    print(f"There are {len(subreddits)} subreddits in the filtered dataset.")

    prolific_authors = get_entries_with_min_submissions(authors, min_submissions=args.author_cutoff)
    print(f"There are {len(prolific_authors)} authors with more than {args.author_cutoff} submissions.")

    active_subreddits = get_entries_with_min_submissions(subreddits, args.subreddit_cutoff)
    print(f"There are {len(active_subreddits)} subreddits with more than {args.subreddit_cutoff} submissions.")

    sorted_active_subreddits = sorted(active_subreddits.items(), key=lambda x: x[1], reverse=True)
    for subreddit, count in sorted_active_subreddits[:5] + sorted_active_subreddits[-5:]:
        print(f"{subreddit:30s}: {count:10d}")

    write_csv(file_path=args.output_authors, data=prolific_authors, column_titles=["author", "count"])
    write_csv(file_path=args.output_subreddits, data=active_subreddits, column_titles=["subreddit", "count"])


if __name__ == "__main__":
    main()

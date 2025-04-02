"""Collect summary statistics for Reddit data: Most prolific authors and active subreddits."""

import collections
import csv
import json

import tqdm

from data_handling import reddit_consts


def get_author_and_subreddit_counts() -> tuple[dict[str, int], dict[str, int]]:
    """Returns the counts of author submissions and subreddit submissions in the filtered dataset."""
    # This function takes about 3 minutes to run.
    authors: dict[str, int] = collections.Counter()
    subreddits: dict[str, int] = collections.Counter()
    with open("../data/RS_2024-05_filtered", "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
            submission = json.loads(line)
            authors[submission["author"]] += 1
            subreddits[submission["subreddit"]] += 1
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
    authors, subreddits = get_author_and_subreddit_counts()
    print(f"There are {len(authors)} authors in the filtered dataset.")
    print(f"There are {len(subreddits)} subreddits in the filtered dataset.")

    author_cutoff = 30
    prolific_authors = get_entries_with_min_submissions(authors, min_submissions=author_cutoff)
    print(f"There are {len(prolific_authors)} authors with more than {author_cutoff} submissions.")

    subreddit_cutoff = 30
    active_subreddits = get_entries_with_min_submissions(subreddits, subreddit_cutoff)
    print(f"There are {len(active_subreddits)} subreddits with more than {subreddit_cutoff} submissions.")

    sorted_active_subreddits = sorted(active_subreddits.items(), key=lambda x: x[1], reverse=True)
    for subreddit, count in sorted_active_subreddits[:5] + sorted_active_subreddits[-5:]:
        print(f"{subreddit:30s}: {count:10d}")

    write_csv(file_path=reddit_consts.PROLIFIC_AUTHORS_FILE, data=prolific_authors, column_titles=["author", "count"])
    write_csv(
        file_path=reddit_consts.ACTIVE_SUBREDDITS_FILE, data=active_subreddits, column_titles=["subreddit", "count"]
    )


if __name__ == "__main__":
    main()

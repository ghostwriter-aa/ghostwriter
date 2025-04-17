"""Filter the Reddit dataset:
1. Remove NSFW subreddits.
2. Restrict to users with more than 10 submissions.
3. Remove some bot users.

This reduces the number of authors by 99% (40M -> 250k),
and reduces the output file size from 187GB to 39GB.
"""

import argparse
import collections
import json

import tqdm

from data_handling import reddit_consts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter the Reddit dataset by removing NSFW subreddits, inactive users, and bots."
    )
    parser.add_argument(
        "--input_submissions",
        type=str,
        default="../data/RS_2024-05",
        required=False,
        help="Path to the input submissions file (e.g., ../data/RS_2024-05).",
    )
    parser.add_argument(
        "--input_comments",
        type=str,
        default="../data/RC_2024-05",
        required=False,
        help="Path to the input comments file (e.g., ../data/RC_2024-05).",
    )
    parser.add_argument(
        "--output_submissions",
        type=str,
        required=True,
        help="Path to the output filtered submissions file (e.g., ../data/RS_2024-05_filtered).",
    )
    parser.add_argument(
        "--output_comments",
        type=str,
        required=True,
        help="Path to the output filtered comments file (e.g., ../data/RC_2024-05_filtered).",
    )
    parser.add_argument(
        "--min_submissions_per_author",
        type=int,
        default=10,
        help="Minimum number of submissions required for an author to be included.",
    )
    args = parser.parse_args()

    # Count per-subreddit over-18 and non-over-18 submissions.
    # We filter out over-18 submissions, but only in subreddits where such submissions are common.
    # This is to avoid users excluding themselves from analysis by arbitrarily marking their posts as over-18.
    subreddit_to_over18_posts: dict[str, int] = collections.Counter()
    subreddit_to_non_over18_posts: dict[str, int] = collections.Counter()
    print("Building over-18 filter...")
    with open(args.input_submissions, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_SUBMISSIONS):
            submission = json.loads(line)
            if submission["over_18"]:
                subreddit_to_over18_posts[submission["subreddit"]] += 1
            else:
                subreddit_to_non_over18_posts[submission["subreddit"]] += 1

    # Count per-author submissions
    authors: dict[str, int] = collections.Counter()
    n_filtered_submissions = 0
    print("Counting author posts...")
    with open(args.input_submissions, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_SUBMISSIONS):
            submission = json.loads(line)
            if submission["author"] in reddit_consts.USERS_TO_IGNORE:
                continue
            subreddit = submission["subreddit"]
            subreddit_over18_ratio = subreddit_to_over18_posts.get(subreddit, 0) / (
                subreddit_to_over18_posts.get(subreddit, 0) + subreddit_to_non_over18_posts.get(subreddit, 0) + 0.1
            )
            if submission.get("over_18") and subreddit_over18_ratio > 0.1:
                continue
            n_filtered_submissions += 1
            authors[submission["author"]] += 1

    if n_filtered_submissions != reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS:
        print(
            f"WARNING! Expected {reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS} filtered submissions, "
            f"got {n_filtered_submissions}."
        )

    prolific_authors = set()
    for author, count in authors.items():
        if count > args.min_submissions_per_author:
            prolific_authors.add(author)

    print(f"There are {len(prolific_authors)} authors with more than {args.min_submissions_per_author} submissions.")

    print("Writing filtered submissions...")
    with open(args.output_submissions, "wt") as outfile:
        with open(args.input_submissions, "rt") as infile:
            for line in tqdm.tqdm(infile, total=reddit_consts.RS_2024_05_SUBMISSIONS):
                submission = json.loads(line)
                if not submission.get("over_18", False) and submission["author"] in prolific_authors:
                    outfile.write(line)

    print("Writing filtered comments...")
    with open(args.output_comments, "wt") as outfile:
        with open(args.input_comments, "rt") as infile:
            for line in tqdm.tqdm(infile, total=reddit_consts.RC_2024_05_COMMENTS):
                comment = json.loads(line)
                if comment["author"] in prolific_authors:
                    outfile.write(line)

    print("Done.")


if __name__ == "__main__":
    main()

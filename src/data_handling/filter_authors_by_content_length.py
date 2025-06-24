"""Filter authors based on their total content length across all subreddits."""

import argparse
import collections
import json

import tqdm


from data_handling import reddit_consts


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter authors based on their total content length.")
    parser.add_argument(
        "--input_submissions",
        type=str,
        required=True,
        help="Path to the input filtered submissions file.",
    )
    parser.add_argument(
        "--input_comments",
        type=str,
        required=True,
        help="Path to the input filtered comments file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file containing prolific authors.",
    )
    parser.add_argument(
        "--output_submissions",
        type=str,
        required=True,
        help="Path to the output filtered submissions file for prolific authors.",
    )
    parser.add_argument(
        "--output_comments",
        type=str,
        required=True,
        help="Path to the output filtered comments file for prolific authors.",
    )
    parser.add_argument(
        "--output_author_and_subreddit_to_stats",
        type=str,
        required=True,
        help="Path to the output author and subreddit to num characters file.",
    )
    parser.add_argument(
        "--min_characters",
        type=int,
        default=10000,
        help="Minimum number of characters required for an author to be included.",
    )
    args = parser.parse_args()

    author_to_subreddit_to_num_characters: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    print("Counting author posts...")
    with open(args.input_submissions, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
            submission = json.loads(line)
            subreddit = submission["subreddit"]
            author_to_subreddit_to_num_characters[submission["author"]][subreddit] += len(submission["selftext"]) + len(
                submission["title"]
            )

    print("Counting author comments...")
    with open(args.input_comments, "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RC_2024_05_FILTERED_COMMENTS):
            comment = json.loads(line)
            subreddit = comment["subreddit"]
            author_to_subreddit_to_num_characters[comment["author"]][subreddit] += len(comment["body"])

    # Calculate total characters per author
    author_to_total_chars: dict[str, int] = {}
    print("Calculating total characters per author...")
    for author in tqdm.tqdm(author_to_subreddit_to_num_characters.keys()):
        author_to_total_chars[author] = sum(author_to_subreddit_to_num_characters[author].values())

    # Filter prolific authors who have at least min_characters in at least two subreddits
    prolific_authors = set()
    for author, subreddit_chars in author_to_subreddit_to_num_characters.items():
        num_subreddits_with_min_chars = sum(1 for chars in subreddit_chars.values() if chars >= args.min_characters)
        if num_subreddits_with_min_chars >= 2:
            prolific_authors.add(author)

    print(f"Found {len(prolific_authors)} authors with at least {args.min_characters} characters")

    # Write prolific authors to output file
    with open(args.output_file, "wt") as outfile:
        for author in sorted(prolific_authors):
            outfile.write(f"{author}\n")

    # Initialize counters. These are for prolific authors only.
    author_to_subreddit_to_num_comments: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    author_to_subreddit_to_num_posts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    print("Writing filtered submissions for prolific authors...")
    with open(args.output_submissions, "wt") as outfile:
        with open(args.input_submissions, "rt") as infile:
            for line in tqdm.tqdm(infile, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
                submission = json.loads(line)
                if submission["author"] in prolific_authors:
                    subreddit = submission["subreddit"]
                    author_to_subreddit_to_num_posts[submission["author"]][subreddit] += 1
                    outfile.write(line)

    print("Writing filtered comments for prolific authors...")
    with open(args.output_comments, "wt") as outfile:
        with open(args.input_comments, "rt") as infile:
            for line in tqdm.tqdm(infile, total=reddit_consts.RC_2024_05_FILTERED_COMMENTS):
                comment = json.loads(line)
                if comment["author"] in prolific_authors:
                    subreddit = comment["subreddit"]
                    author_to_subreddit_to_num_comments[comment["author"]][subreddit] += 1
                    outfile.write(line)

    print("Writing author and subreddit statistics for prolific authors...")
    with open(args.output_author_and_subreddit_to_stats, "wt") as outfile:
        for author in tqdm.tqdm(author_to_subreddit_to_num_characters.keys(), desc="Writing author statistics"):
            if author not in prolific_authors:
                continue
            for subreddit in author_to_subreddit_to_num_characters[author]:
                outfile.write(
                    json.dumps(
                        {
                            "author": author,
                            "subreddit": subreddit,
                            "num_characters": author_to_subreddit_to_num_characters[author][subreddit],
                            "num_comments": author_to_subreddit_to_num_comments[author][subreddit],
                            "num_posts": author_to_subreddit_to_num_posts[author][subreddit],
                        }
                    )
                    + "\n"
                )

    print("Done.")


if __name__ == "__main__":
    main()

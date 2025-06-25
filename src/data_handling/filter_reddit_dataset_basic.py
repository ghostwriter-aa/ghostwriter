"""Filter the Reddit dataset:
1. Remove NSFW subreddits.
2. Remove some bot users.
3. Remove unused keys from the original json data.
"""

import argparse
import collections
import json
import os

import tqdm

from data_handling import reddit_consts


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter the Reddit dataset by removing NSFW subreddits and bots.")
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

    nsfw_group = parser.add_mutually_exclusive_group(required=True)
    nsfw_group.add_argument(
        "--input_nsfw_filter",
        type=str,
        help="Path to the input NSFW filter file. If provided, will use this file instead of calculating a new filter. "
        "Structure of the file: {subreddit_name: over18_ratio} dictionary.",
    )
    nsfw_group.add_argument(
        "--output_nsfw_filter",
        type=str,
        help="Path to save the NSFW filter file e.g. ../summary_data/nsfw_filter.json."
        "Required if --input_nsfw_filter is not provided.",
    )

    args = parser.parse_args()

    # Load or calculate the NSFW filter
    if args.input_nsfw_filter:
        if not os.path.exists(args.input_nsfw_filter):
            print(f"Error: Input NSFW filter file not found at {args.input_nsfw_filter}")
            return
        print("Loading NSFW filter from file...")
        with open(args.input_nsfw_filter, "rt") as f:
            subreddit_to_over18 = json.load(f)
    else:
        # Count per-subreddit over-18 and non-over-18 submissions.
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

        print("Calculating over-18 ratio...")
        subreddit_to_over18 = {}
        for subreddit in set(subreddit_to_over18_posts.keys()) | set(subreddit_to_non_over18_posts.keys()):
            subreddit_to_over18[subreddit] = (
                subreddit_to_over18_posts.get(subreddit, 0)
                / (subreddit_to_over18_posts.get(subreddit, 0) + subreddit_to_non_over18_posts.get(subreddit, 0) + 0.1)
            ) > 0.1

        print(f"Saving NSFW filter to {args.output_nsfw_filter}...")
        os.makedirs(os.path.dirname(args.output_nsfw_filter), exist_ok=True)
        with open(args.output_nsfw_filter, "wt") as f:
            json.dump(subreddit_to_over18, f)

    # When filtering submissions, we also only save fields that are needed in the future to save disk space.
    print("Writing filtered submissions...")
    with open(args.output_submissions, "wt") as outfile:
        with open(args.input_submissions, "rt") as infile:
            for line in tqdm.tqdm(infile, total=reddit_consts.RS_2024_05_SUBMISSIONS):
                submission = json.loads(line)
                if submission["author"] in reddit_consts.USERS_TO_IGNORE:
                    continue
                # We filter out over-18 submissions, but only in subreddits where such submissions are common. This is
                # to avoid users excluding themselves from analysis by arbitrarily marking their posts as over-18.
                if submission.get("over_18", False) and subreddit_to_over18[submission["subreddit"]]:
                    continue
                filtered_submission = {
                    key: submission[key]
                    for key in [
                        "author",
                        "id",
                        "subreddit",
                        "subreddit_id",
                        "subreddit_name_prefixed",
                        "selftext",
                        "title",
                        "over_18",
                        "url",
                    ]
                }
                outfile.write(json.dumps(filtered_submission) + "\n")
    print("Writing filtered comments...")
    with open(args.output_comments, "wt") as outfile:
        with open(args.input_comments, "rt") as infile:
            for line in tqdm.tqdm(infile, total=reddit_consts.RC_2024_05_COMMENTS):
                comment = json.loads(line)
                if comment["author"] not in reddit_consts.USERS_TO_IGNORE:
                    # note: there is no over_18 field in comments
                    filtered_comment = {
                        key: comment[key]
                        for key in [
                            "author",
                            "body",
                            "id",
                            "subreddit",
                            "subreddit_id",
                            "subreddit_name_prefixed",
                            "permalink",
                        ]
                    }
                    outfile.write(json.dumps(filtered_comment) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()

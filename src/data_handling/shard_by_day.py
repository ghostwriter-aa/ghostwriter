import datetime
import json
from typing import Literal

import reddit_consts
import tqdm


def shard_by_day(file_prefix: Literal["RS", "RC"], file_suffix: str = "") -> None:
    day_files = {}
    for day in range(1, 32):
        day_files[day] = open(f"../data/{file_prefix}_2024-05-{day:02d}{file_suffix}", "wt")
    with open(f"../data/{file_prefix}_2024-05{file_suffix}", "rt") as f:
        for line in tqdm.tqdm(f, total=reddit_consts.RS_2024_05_FILTERED_SUBMISSIONS):
            submission = json.loads(line)
            creation_time = datetime.datetime.fromtimestamp(submission["created_utc"], tz=datetime.timezone.utc)
            day_files[creation_time.day].write(line)
    for day_file in day_files.values():
        day_file.close()


if __name__ == "__main__":
    shard_by_day("RS", "_filtered")
    shard_by_day("RC")

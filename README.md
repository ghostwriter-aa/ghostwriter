# Ghostwriter

This project attempts to attribute authorship of text to specific authors. The dataset used is a collection of Reddit posts and comments, with the goal of attributing posts to authors.

The commands below are to be run from the `src/` directory.

## Building the dataset

1. Install the package: From the projects directory, run

   ```
   pip install -e .
   ```

2. Download the Reddit data.

   The data is aggregated into 1-month files. We have thus far downloaded and used only the 2024-05 month. This will give you the files `RS_2024-05` (submissions) and `RC_2024-05` (comments), which you should place in the `../data` directory (_not_ under `/src`).

3. Create an empty directory `../summary_data` (not under `/src`).

4. Run `data_handling/filter_reddit_dataset_basic.py`:

   To calculate and save a new NSFW filter:

   ```bash
   python data_handling/filter_reddit_dataset_basic.py \
       --input_submissions=../data/RS_2024-05 \
       --input_comments=../data/RC_2024-05 \
       --output_submissions=../data/RS_2024-05_filtered \
       --output_comments=../data/RC_2024-05_filtered \
       --output_nsfw_filter=../summary_data/nsfw_filter.json
   ```

   Or to use an existing NSFW filter:

   ```bash
   python data_handling/filter_reddit_dataset_basic.py \
       --input_submissions=../data/RS_2024-05 \
       --input_comments=../data/RC_2024-05 \
       --output_submissions=../data/RS_2024-05_filtered \
       --output_comments=../data/RC_2024-05_filtered \
       --input_nsfw_filter=../summary_data/nsfw_filter.json
   ```

   This script will:

   - Either calculate and save a new NSFW filter or use an existing one
   - Save a filtered version of the submissions and comments to `../data/RS_2024-05_filtered` and `../data/RC_2024-05_filtered`:
     - Filter out NSFW posts in subreddits where such posts are common
     - Filter out submissions and comments from highly active bots
     - Only keep fields that change and are needed (i.e. keep "author", "subreddit", "title", etc.)

   For the reddit May 2024 dataset, the resulting files will be ~15 GB for submissions and ~100 GB for comments.

5. Run `data_handling/filter_authors_by_content_length.py`:

   ```bash
   python data_handling/filter_authors_by_content_length.py \
       --input_submissions=../data/RS_2024-05_filtered \
       --input_comments=../data/RC_2024-05_filtered \
       --output_file=../data/authors_at_least_20_000_characters.jsonl \
       --output_submissions=../data/RS_2024-05_filtered_prolific_authors \
       --output_comments=../data/RC_2024-05_filtered_prolific_authors \
       --output_author_and_subreddit_to_stats=../data/author_and_subreddit_to_stats.jsonl \
       --min_characters=10000
   ```

   This script:

   - Counts the number of characters, comments, and posts per author and subreddit
   - Filters authors who have written at least the minimum number of characters in at least two subreddits, referred to as "prolific authors"
   - Creates filtered submission and comment files containing only prolific authors
   - Saves statistics about character counts, comments, and posts per author and subreddit

6. Run `data_handling/build_summary_data.py`:

   ```bash
   python data_handling/build_summary_data.py \
       --input_submissions=../data/RS_2024-05_filtered \
       --input_comments=../data/RC_2024-05_filtered \
       --output_authors=../summary_data/prolific_authors_cutoff_30.csv \
       --output_subreddits=../summary_data/active_subreddits_cutoff_30.csv \
       --author_cutoff=30 \
       --subreddit_cutoff=30
   ```

   This will create two csv files:

   - `../summary_data/active_subreddits_cutoff_30.csv`, containing the names of subreddits with at least 30 submissions, and the number of submissions in those subreddits.
   - `../summary_data/prolific_authors_cutoff_30.csv`, containing the names of authors with at least 30 submissions, and the number of submissions they posted.

7. Run:

   ```bash
   python data_handling/build_author_docs_dataset.py \
       --input_submissions=../data/RS_2024-05_filtered_prolific_authors \
       --input_comments=../data/RC_2024-05_filtered_prolific_authors \
       --input_active_subreddits=../summary_data/active_subreddits_cutoff_30.csv \
       --input_author_and_subreddit_to_stats=../data/author_and_subreddit_to_stats.jsonl \
       --input_prolific_authors=../data/authors_at_least_20_000_characters.jsonl \
       --output_file=../data/suitable_author_infos.ndjson
   ```

   This will create an NDJSON file where each line is loadable into a `common_types.AuthorInfo` object. Each entry contains the name of an author with at least 10000 characters in two very different subreddits, as well as the names of those subreddits, and all the user's submissions and comments in those two subreddits.

8. Run:

   ```bash
   python data_handling/train_test_split.py \
       --input_file=../data/suitable_author_infos.ndjson \
       --output_prefix=../data/suitable_author_infos
   ```

   This will split `suitable_author_infos.ndjson` into three files for the train, validation, and test arms (`../data/suitable_author_infos_train.ndjson`, etc.).

## Training a model

See the instructions for [training the baseline model](src/baseline_model/README.md) or for [training the embedding model](src/contrastive_learning/README.md).

## Running inference

To check the output of a model on a given set of author pairs, use the inference pipeline:

```bash
python inference/run_inference.py \
    --model_file=../my_model_dir/my_model_file.json \
    --author_docs_file=../data/author_docs.parquet \
    --pairs_to_match_file=../data/required_matches.parquet \
    --output_file=../data/my_output_file.parquet
```

The arguments to this command are:

- `--model_file`: JSON file containing model parameters, built by `build_log_likelihood_model.py`.
- `--author_docs_file`: Parquet file containing the columns `entity` (a string entity identifier), `persona` (a string persona identifier), and `text` (the string containing the document). Any additional columns are ignored.
- `--pairs_to_match_file`: Parquet file containing the columns `entity1`, `persona1`, `entity2`, and `persona2`, representing the IDs of pairs for which inference is requested. Any additional columns are ignored.
- `--output_file`: The name of a Parquet file which will be created, containing the columns `entity1`, `persona1`, `entity2`, and `persona2` from `--pairs_to_match_file`, along with a column `match_score`, a floating-point, model-dependent value indicating the similarity between the two personas (higher values indicate a better match for the log likelihood model, for example).

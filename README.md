# Ghostwriter

This project attempts to attribute authorship of text to specific authors. The dataset used is a collection of Reddit posts and comments, with the goal of attributing posts to authors.

The commands below are to be run from the `src/` directory.

## Steps for building the dataset

1. Install the package: From the projects directory, run

   ```
   pip install -e .
   ```

2. Download the Reddit data.

   The data is aggregated into 1-month files. We have thus far downloaded and used only the 2024-05 month. This will give you the files `RS_2024-05` (submissions) and `RC_2024-05` (comments), which you should place in the `../data` directory (_not_ under `/src`).

3. Run `data_handling/filter_reddit_dataset.py`:

   ```bash
   python data_handling/filter_reddit_dataset.py \
       --input_submissions=../data/RS_2024-05 \
       --input_comments=../data/RC_2024-05 \
       --output_submissions=../data/RS_2024-05_filtered \
       --output_comments=../data/RC_2024-05_filtered \
       --min_submissions_per_author=10
   ```

   This will create the filtered files specified in the output arguments. These files have the same format as the original data, but filter out NSFW posts in subreddits where such posts are common, highly active bots, and users with fewer than the specified minimum submissions.

4. Create an empty directory `../summary_data` (not under `/src`) and run `data_handling/build_summary_data.py`:

   ```bash
   python data_handling/build_summary_data.py \
       --input_submissions=../data/RS_2024-05_filtered \
       --output_authors=../summary_data/prolific_authors_cutoff_30.csv \
       --output_subreddits=../summary_data/active_subreddits_cutoff_30.csv \
       --author_cutoff=30 \
       --subreddit_cutoff=30
   ```

   This will create two csv files:

   - `../summary_data/active_subreddits_cutoff_30.csv`, containing the names of subreddits with at least 30 submissions, and the number of submissions in those subreddits.
   - `../summary_data/prolific_authors_cutoff_30.csv`, containing the names of authors with at least 30 submissions, and the number of submissions they posted.

5. Run:

   ```bash
   python data_handling/build_author_docs_dataset.py \
       --input_submissions=../data/RS_2024-05_filtered \
       --input_comments=../data/RC_2024-05_filtered \
       --input_active_subreddits=../summary_data/active_subreddits_cutoff_30.csv \
       --output_file=../data/suitable_author_infos.ndjson
   ```

   This will create an NDJSON file where each line is loadable into a `common_types.AuthorInfo` object. Each entry contains the name of an author with at least 5 submissions in two very different subreddits, as well as the names of those subreddits, and all the user's submissions and comments in those two subreddits.

6. Run:

   ```bash
   python data_handling/train_test_split.py \
       --input_file=../data/suitable_author_infos.ndjson \
       --output_prefix=../data/suitable_author_infos
   ```

   This will split `suitable_author_infos.ndjson` into three files for the train, validation, and test arms (`../data/suitable_author_infos_train.ndjson`, etc.).

## Running the baseline model

6. To create the common tokens file, run:

   ```bash
   python baseline_model/get_token_frequencies.py \
       --input_author_infos=../data/suitable_author_infos_train.ndjson \
       --output_file=../data/tiktoken_counts_top_1000.json \
       --top_k=1000 \
       --min_author_occurrences=20
   ```

   This will create a file with the 1000 most common tokens in the dataset, with at least 20 authors using those tokens.

7. Filter authors even further - those who have at least 500 tokens from the 1000 most common ones.

   ```bash
   python baseline_model/filter_authors_according_to_num_of_tokens.py \
       --input_author_infos=../data/suitable_author_infos_train.ndjson \
       --output_file=../data/author_infos_many_tokens_train.ndjson \
       --most_frequent_tokens_file=../data/tiktoken_counts_top_1000.json \
       --sufficient_tokens=500
   ```

   For the validation set:

   ```bash
   python baseline_model/filter_authors_according_to_num_of_tokens.py \
       --input_author_infos=../data/suitable_author_infos_val.ndjson \
       --output_file=../data/author_infos_many_tokens_val.ndjson \
       --most_frequent_tokens_file=../data/tiktoken_counts_top_1000.json \
       --sufficient_tokens=500
   ```

8. Optional: run baseline_model/data_exploration.ipynb to get see a comparison of probabilities for tokens to appear in different personas.

9. Run find_best_tokens.ipynb, which will find the best 1-gram single token classifiers. Those will be saved to a file
   for later use. In the end of it, you will have the base log-likelihood based model, which achieves ±0.79 accuracy,
   using the 40 best tokens ("best" - in the sense of most accurate 1-gram classifiers).

10. Optional: run additional basic models (baseline_model/logistic_regression.ipynb, baseline_model/decision_trees.ipynb), and / or more data exploration (baseline_model/data_exploration_40_best_tokens.ipynb).

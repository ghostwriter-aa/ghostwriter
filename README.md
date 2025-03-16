# Ghostwriter

This project attempts to attribute authorship of text to specific authors.
The dataset used is a collection of Reddit posts and comments, with the goal of attributing posts to authors.

## Steps for building the dataset

1. Install the package: From the projects directory, run

   ```
   pip install -e .
   ```

2. Download the Reddit data.

   The data is aggregated into 1-month files. We have thus far downloaded and used only the 2024-05 month. This will give you the files `RS_2024-05` (submissions) and `RC_2024-05` (comments), which you should place in the `data` directory.

3. Run `filter_reddit_dataset.py`, which will create the filtered files `RS_2024-05_filtered` and `RC_2024-05_filtered`. These files have the same format as the original data, but filter out NSFW posts in subreddits where such posts are common, highly active bots, and users with fewer than 10 submissions per month.

4. Run (from src):

   ```
   python data_handling/build_author_docs_dataset.py --output_file=./../data/suitable_author_infos.ndjson
   ```

   This will create an NDJSON file where each line is loadable into a `common_types.AuthorInfo` object. Each entry contains the name of an author with at least 5 submissions in two very different subreddits, as well as the names of those subreddits, and all the user's submissions and comments in those two subreddits.

5. Run:

   ```
   python data_handling/train_test_split.py \
       --input_file=../data/suitable_author_infos.ndjson \
       --output_prefix=../data/suitable_author_infos
   ```

   This will split `suitable_author_infos.ndjson` into three files for the train, validation, and test arms.

## Running the baseline model

6. To create the common tokens file, run:

   ```
   python baseline_model/get_token_frequencies.py --output_file ../data/tiktoken_counts_top_1000.json --top_k=1000 --min_author_occurrences 20
   ```

   This will create a file with the 1000 most common tokens in the dataset, with at least 20 authors using those tokens.

7. Filter authors even further - those who have at least 500 tokens from the 1000 most common ones.

   ```
   python baseline_model/filter_authors_according_to_num_of_tokens.py \
       --input_author_infos=../data/suitable_author_infos_train.ndjson \
       --output_file=../data/author_infos_many_tokens_train.ndjson \
       --most_frequent_tokens_file=../data/tiktoken_counts_top_1000.json \
       --sufficient_tokens=500
   ```

   For the validation set:

   ```
   python baseline_model/filter_authors_according_to_num_of_tokens.py \
       --input_author_infos=../data/suitable_author_infos_val.ndjson \
       --output_file=../data/author_infos_many_tokens_val.ndjson \
       --most_frequent_tokens_file=../data/tiktoken_counts_top_1000.json \
       --sufficient_tokens=500
   ```

7a. Possible: run baseline_model/data_exploration.ipynb* to get see a comparison of probabilities for tokens to appear in
different personas.

8. Run find_best_tokens.ipynb, which will find the best 1-gram single token classifiers. Those will be saved to a file  
   for later use. In the end of it, you will have the base log-likelihood based model, which achieves ±0.79 accuracy,
   using the 40 best tokens ("best" - in the sense of most accurate 1-gram classifiers).

9. Optional: run additional basic models (base_model/logistic_regression.ipynb, base_model/decision_trees.ipynb), and / or
   more data exploration (base_model/data_exploration_40_best_tokens.ipynb).

*(Note: ipynb are saved as .py files in the repository, and can be converted to notebooks.)
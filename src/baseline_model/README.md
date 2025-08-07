## Training the baseline model

1. To create the common tokens file, run:

   ```bash
   python baseline_model/get_token_frequencies.py \
       --input_author_infos=../data/suitable_author_infos_train.ndjson \
       --output_file=../data/tiktoken_counts_top_1000.json \
       --top_k=1000 \
       --min_author_occurrences=20
   ```

   This will create a file with the 1000 most common tokens in the dataset, with at least 20 authors using those tokens.

2. To create a baseline 1-gram log-likelihood model, run:

   ```bash
   python baseline_model/build_log_likelihood_model.py \
       --num_tokens_to_use=40 \
       --tiktoken_counts_top_1000_file=../data/tiktoken_counts_top_1000.json \
       --suitable_author_infos_train_file=../data/suitable_author_infos_train.ndjson \
       --output_file=../my_model_dir/my_model_file.json
   ```

   This will create a JSON file containing the 40 tokens which are most helpful in distinguishing between author personas. This JSON file is the only thing you need in order to perform inference (see instructions below).

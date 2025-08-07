#!/bin/bash

DECOMPRESSED_DUMPS_DIR=""
TRAIN_DATASET_PATH=""
SELECT_DECOMPRESSED_DUMP_MONTH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --decompressed-dumps-dir)
      DECOMPRESSED_DUMPS_DIR="$2"
      shift 2
      ;;
    --train-dataset-path)
      TRAIN_DATASET_PATH="$2"
      shift 2
      ;;
    --dump-month)
      SELECT_DECOMPRESSED_DUMP_MONTH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --decompressed-dumps-dir PATH --train-dataset-path PATH --dump-month YYYY-MM"
      exit 1
      ;;
  esac
done

# Check that all required arguments are provided
if [[ -z "$DECOMPRESSED_DUMPS_DIR" ]]; then
  echo "Error: --decompressed-dumps-dir is required, got '$DECOMPRESSED_DUMPS_DIR'"
  exit 1
fi

if [[ -z "$TRAIN_DATASET_PATH" ]]; then
  echo "Error: --train-dataset-path is required, got '$TRAIN_DATASET_PATH'"
  exit 1
fi

if [[ -z "$SELECT_DECOMPRESSED_DUMP_MONTH" ]]; then
  echo "Error: --dump-month is required, got '$SELECT_DECOMPRESSED_DUMP_MONTH'"
  exit 1
fi

# Construct paths
SUBMISSIONS_DUMP="RS_${SELECT_DECOMPRESSED_DUMP_MONTH}"
COMMENTS_DUMP="RC_${SELECT_DECOMPRESSED_DUMP_MONTH}"

# Create directories
mkdir -p "${TRAIN_DATASET_PATH}"
SUMMARY_DATA_PATH="${TRAIN_DATASET_PATH}/summary_data"
mkdir -p "${SUMMARY_DATA_PATH}"
BASIC_FILTER_DATASET_PATH="${TRAIN_DATASET_PATH}/basic_filtered_dumps"
mkdir -p "${BASIC_FILTER_DATASET_PATH}"

export SUMMARY_DATA_PATH
export TRAIN_DATASET_PATH
export FULL_SUBMISSIONS="${DECOMPRESSED_DUMPS_DIR}/submissions/${SUBMISSIONS_DUMP}"
export FULL_COMMENTS="${DECOMPRESSED_DUMPS_DIR}/comments/${COMMENTS_DUMP}"
export FILTERED_SUBMISSIONS="${BASIC_FILTER_DATASET_PATH}/${SUBMISSIONS_DUMP}_filtered"
export FILTERED_COMMENTS="${BASIC_FILTER_DATASET_PATH}/${COMMENTS_DUMP}_filtered"
export PROLIFIC_SUBMISSIONS="${TRAIN_DATASET_PATH}/${SUBMISSIONS_DUMP}_filtered_prolific_authors"
export PROLIFIC_COMMENTS="${TRAIN_DATASET_PATH}/${COMMENTS_DUMP}_filtered_prolific_authors"
export PYTHONPATH="src"

echo "Setting up training data..."
echo "Decompressed dumps directory: ${DECOMPRESSED_DUMPS_DIR}"
echo "Train dataset path: ${TRAIN_DATASET_PATH}"
echo "Processing month: ${SELECT_DECOMPRESSED_DUMP_MONTH}"

# Run the processing pipeline
echo "Step 1: Basic filtering..."
python src/data_handling/filter_reddit_dataset_basic.py \
  --input_submissions "$FULL_SUBMISSIONS" \
  --input_comments "$FULL_COMMENTS" \
  --output_submissions "$FILTERED_SUBMISSIONS" \
  --output_comments "$FILTERED_COMMENTS" \
  --output_nsfw_filter="$SUMMARY_DATA_PATH/nsfw_filter.json"

echo "Step 2: Filtering authors by content length..."
python src/data_handling/filter_authors_by_content_length.py \
    --input_submissions="$FILTERED_SUBMISSIONS" \
    --input_comments="$FILTERED_COMMENTS" \
    --output_file="$TRAIN_DATASET_PATH/authors_at_least_20_000_characters.jsonl" \
    --output_submissions="$PROLIFIC_SUBMISSIONS" \
    --output_comments="$PROLIFIC_COMMENTS" \
    --output_author_and_subreddit_to_stats="$TRAIN_DATASET_PATH/author_and_subreddit_to_stats.jsonl" \
    --min_characters=10000

echo "Step 3: Building summary data..."
python src/data_handling/build_summary_data.py \
    --input_submissions="$FILTERED_SUBMISSIONS" \
    --input_comments="$FILTERED_COMMENTS" \
    --output_authors="$SUMMARY_DATA_PATH/prolific_authors_cutoff_30.csv" \
    --output_subreddits="$SUMMARY_DATA_PATH/active_subreddits_cutoff_30.csv" \
    --author_cutoff=30 \
    --subreddit_cutoff=30

echo "Step 4: Building author docs dataset..."
python src/data_handling/build_author_docs_dataset.py \
    --input_submissions="$PROLIFIC_SUBMISSIONS" \
    --input_comments="$PROLIFIC_COMMENTS" \
    --input_active_subreddits="$SUMMARY_DATA_PATH/active_subreddits_cutoff_30.csv" \
    --input_author_and_subreddit_to_stats="$TRAIN_DATASET_PATH/author_and_subreddit_to_stats.jsonl" \
    --input_prolific_authors="$TRAIN_DATASET_PATH/authors_at_least_20_000_characters.jsonl" \
    --output_file="$TRAIN_DATASET_PATH/suitable_author_infos.ndjson"

echo "Step 5: Train/test split..."
python src/data_handling/train_test_split.py \
    --input_file="$TRAIN_DATASET_PATH/suitable_author_infos.ndjson" \
    --output_prefix="$TRAIN_DATASET_PATH/suitable_author_infos"

echo "Step 6: Getting token frequencies..."
python src/baseline_model/get_token_frequencies.py \
    --input_author_infos="$TRAIN_DATASET_PATH/suitable_author_infos_train.ndjson" \
    --output_file="$TRAIN_DATASET_PATH/tiktoken_counts_top_1000.json" \
    --top_k=1000 \
    --min_author_occurrences=20

echo "Step 7: Building log likelihood model..."
python src/baseline_model/build_log_likelihood_model.py \
    --num_tokens_to_use=40 \
    --tiktoken_counts_top_1000_file="$TRAIN_DATASET_PATH/tiktoken_counts_top_1000.json" \
    --suitable_author_infos_train_file="$TRAIN_DATASET_PATH/suitable_author_infos_train.ndjson" \
    --output_file="$SUMMARY_DATA_PATH/log_likelihood_model.json"

echo "Training data setup complete!"
echo "Output files in: ${TRAIN_DATASET_PATH}"
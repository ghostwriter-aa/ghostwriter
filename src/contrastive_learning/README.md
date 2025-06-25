# Instructions for running Contrastive Learning

## Overview

This module contains the code for the contrastive learning model.

## Usage

1. Create vector embeddings for each persona.
   There are two factors that go into this:

   - Embeddeer: The model used to create the embeddings.
     - nomic
     - e5 (base v2)
     - e5-multilingual-large-instruct
   - EmbeddingStrategy: The embedding strategy used to create the embeddings.
     - Using an embedding for each submission, and then averaging the embeddings for each persona.
     - Concatenating the submissions with a delimiter token, and then running an embedding model on the concatenated
       string. (Not implemented)

   This classes for the Embedder and EmbeddingStrategy are in `embedding.py`, and you can embed with `embed_main.py`
   using the following command (from src):

   ```bash
        python3 contrastive_learning/embed_main.py \
        --train-file ./../data/suitable_author_infos_train.ndjson \
        --val-file ./../data/suitable_author_infos_val.ndjson \
        --embedder e5 \
        --output ./../data/embeddings/e5_embedding_{arm}.json
   ```

2. Train the contrastive learning model on the embeddings:

   ```bash
   python3 contrastive_learning/contrastive_learner.py \
       --train-embedding-file ./../data/embeddings/e5_embedding_train.json \
       --val-embedding-file ./../data/embeddings/e5_embedding_val.json \
       --hidden-dim 0 \
       --output-dim 256 \
       --batch-size 64 \
       --max-epochs 15
   ```

   For the training set of about 3700 samples, the above takes about 2 mins for 15 epochs on ec2.g5 instance.

   This will train a SimCLR model that learns to distinguish between different writing styles. The model will be saved
   in the `./../data/checkpoints/TextSimCLR` directory.

3. Visualize and analyze the results:
   Open and run `visualize_results.ipynb` to see performance metrics - accuracy when the prior is 0.5 prob of the same
   author and 0.5 probability of different authors (when given two persona vector embeddings).

4. Optional: Train with
   ```bash
   python3 contrastive_learning/contrastive_learner.py \
       --train-embedding-file ./../data/embeddings/e5_embedding_train.json \
       --val-embedding-file ./../data/embeddings/e5_embedding_val.json \
       --hidden-dim 256 \
       --output-dim 128 \
       --batch-size 64
   ```
   to see the results of a deeper neural network (256 -> 128, fully connected). The results are almost identical.

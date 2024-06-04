#!/usr/bin/env bash

# Download raw wikitext datasets
# ./wikitext.sh

# Convert raw wikitext datasets to plaintext
python -m tok.cli.parquet_to_txt -d data/wikitext-2-raw-v1
python -m tok.cli.parquet_to_txt -d data/wikitext-103-raw-v1

# Train plaintext wikitext datasets using tokenizers
# NOTE: train will be renamed to tokenizers in the future to reduce ambiguity
python -m tok.cli.train -t -d data/wikitext-2-raw-v1
python -m tok.cli.train -t -d data/wikitext-103-raw-v1

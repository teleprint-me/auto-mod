#!/usr/bin/env bash

function wikitext_2_raw_v1() {
    declare -r WIKI_PATH='data/wikitext-2-raw-v1'

    declare -A WIKI_MAP=(
        ["test"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet?download=true'
        ["train"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet?download=true'
        ["validation"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet?download=true'
    )

    declare -A OUTPUT_MAP=(
        ["test"]="${WIKI_PATH}/test-00000-of-00001.parquet"
        ["train"]="${WIKI_PATH}/train-00000-of-00001.parquet"
        ["validation"]="${WIKI_PATH}/validation-00000-of-00001.parquet"
    )

    mkdir -p "${WIKI_PATH}"

    for key in "${!WIKI_MAP[@]}"
    do
    wget "${WIKI_MAP[$key]}" -O "${OUTPUT_MAP[$key]}"
    done
}

function wikitext_103_raw_v1() {
    declare -r WIKI_PATH='data/wikitext-103-raw-v1'

    declare -A WIKI_MAP=(
        ["test"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet?download=true'
        ["train_0"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet?download=true'
        ["train_1"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet?download=true'
        ["validation"]='https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet?download=true'
    )

    declare -A OUTPUT_MAP=(
        ["test"]="${WIKI_PATH}/test-00000-of-00001.parquet"
        ["train_0"]="${WIKI_PATH}/train-00000-of-00002.parquet"
        ["train_1"]="${WIKI_PATH}/train-00001-of-00002.parquet"
        ["validation"]="${WIKI_PATH}/validation-00000-of-00001.parquet"
    )

    mkdir -p "${WIKI_PATH}"

    for key in "${!WIKI_MAP[@]}"
    do
    wget "${WIKI_MAP[$key]}" -O "${OUTPUT_MAP[$key]}"
    done
}

wikitext_2_raw_v1
wikitext_103_raw_v1

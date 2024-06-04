#!/usr/bin/env bash

declare -A URL_MAP=(
    ["zip"]="https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT"
    ["gguf"]="https://raw.githubusercontent.com/ggerganov/ggml/master/docs/gguf.md"
    ["model-cards"]="https://raw.githubusercontent.com/huggingface/hub-docs/main/modelcard.md"
    ["safetensors"]="https://github.com/huggingface/safetensors/raw/main/docs/safetensors.schema.json"
)

declare -A OUTPUT_MAP=(
    ["zip"]="specs/zip.txt"
    ["gguf"]="specs/gguf.md"
    ["model-cards"]="specs/model-cards.md"
    ["safetensors"]="specs/safetensors.json"
)

for key in "${!URL_MAP[@]}"
do
  wget "${URL_MAP[$key]}" -O "${OUTPUT_MAP[$key]}"
done

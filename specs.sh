#!/usr/bin/env bash

mkdir specs
wget https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT -O specs/zip.txt
wget https://raw.githubusercontent.com/ggerganov/ggml/master/docs/gguf.md -O specs/gguf.md
wget https://raw.githubusercontent.com/huggingface/hub-docs/main/modelcard.md -O specs/model-cards.md
wget https://github.com/huggingface/safetensors/raw/main/docs/safetensors.schema.json -O specs/safetensors.json

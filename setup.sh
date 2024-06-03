#!/usr/bin/env bash
#
# setup.sh: Script to setup dev environ
#
if [[ ! -x $(which virtualenv) ]]; then
    echo '"virtualenv" is not installed.';
    exit 1;
fi

virtualenv .venv
source .venv/bin/activate
pip install torch torchtext --index-url https://download.pytorch.org/whl/cpu --upgrade
pip install -r requirements.txt --upgrade

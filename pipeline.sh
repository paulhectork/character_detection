#!/usr/bin/env bash

set -e

if [ ! -d venv ]; then echo "virtual env './venv/' not found. exiting" && exit 1; fi

source venv/bin/activate
python LinePredictor/pipeline_inference.py

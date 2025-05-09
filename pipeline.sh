#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

VENV_DIR="$SCRIPT_DIR/venv"
DATA_DIR="$SCRIPT_DIR"/data
DATA_LP_DIR="$SCRIPT_DIR"/data_line_prediction
DATA_DTLR_DIR="$SCRIPT_DIR"/data_character_detection
LP_DIR="$SCRIPT_DIR"/LinePredictor
DTLR_DIR="$SCRIPT_DIR"/DTLR

if [ ! -d "$VENV_DIR" ]; then echo "virtual env 'venv' not found (at '$VENV_DIR'). exiting" && exit 1; fi

source "$VENV_DIR"/bin/activate

python "$LP_DIR"/pipeline_inference.py\
    -i "$DATA_DIR"\
    -o "$DATA_LP_DIR"

#!/usr/bin/env bash

set -e

# ---------------------------------------
# variables

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

VENV_DIR="$SCRIPT_DIR/venv"
DATA_DIR="$SCRIPT_DIR"/data
DATA_LP_DIR="$SCRIPT_DIR"/data_line_prediction
DATA_DTLR_DIR="$SCRIPT_DIR"/data_character_detection
LP_DIR="$SCRIPT_DIR"/LinePredictor
DTLR_DIR="$SCRIPT_DIR"/DTLR

SAMPLE=False

# ---------------------------------------
# functions

print_usage() {
    cat<<EOF

USAGE bash pipeline.sh [-s]

    run the character detection pipeline

    -s : run the pipeline on a sample of 10 images (instead of 6000+ images)

EOF
}

# ---------------------------------------
# process

while getopts 's' flag; do
    case "${flag}" in
        s) SAMPLE=True;;
        *) print_usage
           exit 1;;
    esac
done

if [ ! -d "$VENV_DIR" ]; then echo "virtual env 'venv' not found (at '$VENV_DIR'). exiting" && exit 1; fi

source "$VENV_DIR"/bin/activate

# step 1: character detection
if "$SAMPLE"; then
    python "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -s
else
    python "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"
fi

#TODO step 2

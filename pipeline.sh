#!/usr/bin/env bash

set -e

# ---------------------------------------
# variables

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

VENV_DIR="$SCRIPT_DIR"/venv
DATA_DIR="$SCRIPT_DIR"/data
DATA_LP_DIR="$SCRIPT_DIR"/data_line_prediction
DATA_DTLR_DIR="$SCRIPT_DIR"/data_character_detection
LP_DIR="$SCRIPT_DIR"/LinePredictor
DTLR_DIR="$SCRIPT_DIR"/DTLR

SAMPLE=false
VISUALIZE=false

# ---------------------------------------
# functions

print_usage() {
    cat<<EOF

USAGE bash pipeline.sh [-s]

    run the character detection pipeline

    -s : run the pipeline on a sample of 10 images (instead of 6000+ images)
    -v: produce visualizations instead of saving bounding boxes JSON (will process only the first 10 images)

EOF
}

# ---------------------------------------
# process

while getopts 'svh' flag; do
    case "${flag}" in
        s) SAMPLE=true;;
        v) VISUALIZE=true
           SAMPLE=true;;
        h) print_usage
           exit 0;;
        *) print_usage
           exit 1;;
    esac
done

if [ ! -d "$VENV_DIR" ]; then echo "virtual env 'venv' not found (at '$VENV_DIR'). exiting" && exit 1; fi

source "$VENV_DIR"/bin/activate

if "$VISUALIZE"; then
    # step 1
    python "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -s
    python "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -v
    # step 2
    python "$DTLR_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -b "$DATA_LP_DIR"\
        -o "$DATA_DTLR_DIR"
        -v
elif "$SAMPLE"; then
    python "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -s
    # step 2
    python "$DTLR_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -b "$DATA_LP_DIR"\
        -o "$DATA_DTLR_DIR"\
        -s
else
    python "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"
    # step 2
    python "$DTLR_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -b "$DATA_LP_DIR"\
        -o "$DATA_DTLR_DIR"
fi


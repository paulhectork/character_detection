#!/usr/bin/env bash

set -e

# ---------------------------------------
# variables

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATA_DIR="$SCRIPT_DIR"/data
DATA_LP_DIR="$SCRIPT_DIR"/data_line_prediction
DATA_DTLR_DIR="$SCRIPT_DIR"/data_character_detection
LP_DIR="$SCRIPT_DIR"/LinePredictor
DTLR_DIR="$SCRIPT_DIR"/DTLR

SAMPLE=false
VISUALIZE=false

UV="uv run"
# --directory='$SCRIPT_DIR'"

# ---------------------------------------
# functions

print_usage() {
    cat<<EOF

USAGE bash pipeline.sh [-s|-v]?

    run the character detection pipeline

    -s : run the pipeline on a sample of 10 images (instead of 6000+ images)
    -v : produce visualizations instead of saving bounding boxes JSON (will process only the first 10 images)

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

if "$VISUALIZE"; then
    # step 1
    $UV "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -s
    $UV "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -v
    # step 2
    $UV "$DTLR_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -b "$DATA_LP_DIR"\
        -o "$DATA_DTLR_DIR"\
        -v
elif "$SAMPLE"; then
    $UV "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"\
        -s
    # step 2
    $UV "$DTLR_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -b "$DATA_LP_DIR"\
        -o "$DATA_DTLR_DIR"\
        -s
else
    $UV "$LP_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -o "$DATA_LP_DIR"
    # step 2
    $UV "$DTLR_DIR"/pipeline_inference.py\
        -i "$DATA_DIR"\
        -b "$DATA_LP_DIR"\
        -o "$DATA_DTLR_DIR"
fi

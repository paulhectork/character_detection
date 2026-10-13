#!/usr/bin/env bash

# ---------------------
# variables

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LP_DIR="$SCRIPT_DIR"/LinePredictor
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

# ---------------------
# process

if ! command -v uv >/dev/null 2>&1
then
    echo "uv is required. exiting..."
    exit 1
fi

# init submodules
git submodule init && git submodule sync --recursive

# install packages
uv sync

# compile dino cuda ops
cd "$LP_DIR"/models
"$PYTHON_BIN" ./dino/ops/setup.py build install



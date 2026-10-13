# CHARACTER DETECTION

A character detection pipeline based on [DTLR](https://github.com/raphael-baena/DTLR). Most of the work is done in the submodules.

---

## STRUCTURE

```
root/
 |_data                       # input files go here
 |_data_line_prediction       # output from step 1
 |_data_character_detection   # output from step 2
 |_DTLR                       # step 2: character detection
 |_LinePredictor              # step 1: line detection
```

---

## SETUP

### Clone

```bash
git clone git@github.com:paulhectork/character_detection.git
cd submodule
git submodule init && git submodule update
```

### Setup

Requires CUDA and UV. The installation script will init submodules, install python deps, compile DINO CUDA operators.

```bash
bash install.sh
```

---

## USE

Use the scripted pipeline:

```bash
bash pipeline.sh [-s|-v]
```

Or do it manually:

```bash
source .venv/bin/activate

# step 1: line dedection
python LinePredictor/pipeline_inference.py -i ./data -o ./data_line_prediction/ [-s|-v]

# step 2: character detection
python DTLR/pipeline_inference.py -i ./data -b ./data_line_prediction -o ./data_character_detection [-s|-v]
```


# CHARACTER PREDICTION

## STRUCTURE

```
root/
 |_DTLR           # step 2: character detection
 |_LinePredictor  # step 1: line detection
```

## SETUP

To generate a requirements file combining requirements from the submodules, you can use [fuser](https://github.com/paulhectork/fuser)

1. Setup CUDA

	```bash
	nvcc --version  # displays cuda version currently in use
	nvidia-smi      # on the top-right, displays higher version installed on the system

	# make sure nvcc version matches selected CUDA_HOME
	export CUDA_HOME=<path/to/cuda>   # often `/usr/local/cuda`a
	export PATH=$CUDA_HOME/bin:$PATH
	export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
	```
2. Setup python venv
	```bash
	python3 -m venv venv
	source venv/bin/activate
	pip install wheel
	pip install -r requirements.txt
	```
3. Compile CUDA operators
	```bash
	cd LinePredictor/models/
	python ./dino/ops/setup.py build install
	# Unit test => could output an outofmemory error
	python ./dino/ops/test.py
	```


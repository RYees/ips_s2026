SDK_ROOT := ../../../pyorbbecsdk
VENV := venv/bin/python3
PYTHONPATH := $(abspath $(SDK_ROOT)/install/lib)
LD_LIBRARY_PATH := $(abspath $(SDK_ROOT)/install/lib)

.PHONY: run env run-preview run-app

env:
	@echo "Setting Orbbec SDK environment: $(PYTHONPATH)"
	@echo "Use this for shell sessions:"
	@echo "  export PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH"
	@echo "  export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH"

run:
	@echo "Running main.py with Orbbec SDK environment..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) main.py

run-preview:
	@echo "Running camera preview tool with Orbbec SDK environment..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) tools/camera_preview.py

run-app: run-preview

PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/cpsstudent/Documents/pyorbbecsdk/install/python \
LD_LIBRARY_PATH=/home/cpsstudent/Documents/pyorbbecsdk/install/lib \
python3 main.py
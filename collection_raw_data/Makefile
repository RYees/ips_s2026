SDK_ROOT := ../../pyorbbecsdk
VENV := /home/cpsstudent/Documents/.venv/bin/python 
PYTHONPATH := $(abspath $(SDK_ROOT)/install/lib)
LD_LIBRARY_PATH := $(abspath $(SDK_ROOT)/install/lib)

.PHONY: run env run-preview run-app sample masks

env:
	@echo "Setting Orbbec SDK environment: $(PYTHONPATH)"
	@echo "Use this for shell sessions:"
	@echo "  export PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH"
	@echo "  export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH"

run:
	@echo "Running main.py with Orbbec SDK environment..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) main.py

demo:
	@echo "Launching real-time sorting system with Orbbec camera..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) live/live_inference.py

demo1:
	@echo "Launching real-time sorting system with Orbbec camera..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) live/live.py

run-preview:
	@echo "Running camera preview tool with Orbbec SDK environment..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) tools/camera_preview.py

debug:
	@echo "Running camera preview tool with Orbbec SDK environment..."
	PYTHONPATH=$(PYTHONPATH):$$PYTHONPATH LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$$LD_LIBRARY_PATH $(VENV) processing/debug_projection.py


run-app: run-preview

PYTHON ?= $(VENV)
SAMPLE_STEM ?= img0935
SAMPLE_SAMPLES_ROOT ?= ../offline_case/samples
SAMPLE_OUTPUT_DIR ?= ../offline_case/output/$(SAMPLE_STEM)
SAMPLE_INTRINSICS_FILE ?=
SAMPLE_CLI_INTRINSICS ?=
RAW_DATA_ROOT ?= data
RAW_MASK_STEM ?=
RAW_MASK_OVERWRITE ?=

sample:
	@echo "Running offline sample evaluator for $(SAMPLE_STEM)..."
	$(PYTHON) ../offline_case/evaluate_case.py \
		--samples-root $(SAMPLE_SAMPLES_ROOT) \
		--stem $(SAMPLE_STEM) \
		--output-dir $(SAMPLE_OUTPUT_DIR) \
		$(if $(strip $(SAMPLE_INTRINSICS_FILE)),--intrinsics-file $(SAMPLE_INTRINSICS_FILE),) \
		$(if $(strip $(SAMPLE_CLI_INTRINSICS)),--width $(word 1,$(SAMPLE_CLI_INTRINSICS)) --height $(word 2,$(SAMPLE_CLI_INTRINSICS)) --fx $(word 3,$(SAMPLE_CLI_INTRINSICS)) --fy $(word 4,$(SAMPLE_CLI_INTRINSICS)) --cx $(word 5,$(SAMPLE_CLI_INTRINSICS)) --cy $(word 6,$(SAMPLE_CLI_INTRINSICS)),)

masks:
	@echo "Generating raw masks from $(RAW_DATA_ROOT)..."
	$(PYTHON) tools/generate_masks_from_raw.py \
		--dataset-root $(RAW_DATA_ROOT) \
		$(if $(strip $(RAW_MASK_STEM)),--stem $(RAW_MASK_STEM),) \
		$(if $(strip $(RAW_MASK_OVERWRITE)),--overwrite,)

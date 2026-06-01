PYTHON ?= python3
SAMPLE_STEM ?= img0000
SAMPLE_SAMPLES_ROOT ?= offline_case/samples
SAMPLE_OUTPUT_DIR ?= offline_case/output/$(SAMPLE_STEM)
SAMPLE_INTRINSICS_FILE ?=
SAMPLE_CLI_INTRINSICS ?=
DATASET_ROOT ?= offline_case/dataset
DATASET_IMAGES_DIR ?= $(DATASET_ROOT)/images
DATASET_MASKS_DIR ?= $(DATASET_ROOT)/masks
DATASET_LABELS_DIR ?= $(DATASET_ROOT)/labels

.PHONY: sample

sample:
	@echo "Running offline sample evaluator for $(SAMPLE_STEM)..."
	$(PYTHON) offline_case/evaluate_case.py \
		--samples-root $(SAMPLE_SAMPLES_ROOT) \
		--stem $(SAMPLE_STEM) \
		--output-dir $(SAMPLE_OUTPUT_DIR) \
		$(if $(strip $(SAMPLE_INTRINSICS_FILE)),--intrinsics-file $(SAMPLE_INTRINSICS_FILE),) \
		$(if $(strip $(SAMPLE_CLI_INTRINSICS)),--width $(word 1,$(SAMPLE_CLI_INTRINSICS)) --height $(word 2,$(SAMPLE_CLI_INTRINSICS)) --fx $(word 3,$(SAMPLE_CLI_INTRINSICS)) --fy $(word 4,$(SAMPLE_CLI_INTRINSICS)) --cx $(word 5,$(SAMPLE_CLI_INTRINSICS)) --cy $(word 6,$(SAMPLE_CLI_INTRINSICS)),)


TEST_STEM ?= img0018
TEST_BASE ?= offline_case/samples

.PHONY: test

test:
	@echo "Running offline visualization test for $(TEST_STEM)..."
	$(PYTHON) offline_case/test.py \
		$(TEST_STEM) \
		--base $(TEST_BASE)

mask:
# 	@echo "Generating object mask for $(TEST_STEM)..."
# 	$(PYTHON) offline_case/pointcloud_to_mask.py \
# 		$(TEST_STEM) \
# 		--base $(TEST_BASE)
	@echo "Generating object mask for $(TEST_STEM)..."
	$(PYTHON) offline_case/masking.py \
		$(TEST_STEM) \
		--base $(TEST_BASE)

.PHONY: maskall

maskall:
	@echo "Generating object masks for all samples in $(DATASET_ROOT)..."
	@set -e; \
	for img in $(sort $(wildcard $(DATASET_IMAGES_DIR)/*.png)); do \
		stem=$$(basename "$$img" .png); \
		echo "Generating object mask for $$stem..."; \
		$(PYTHON) offline_case/masking.py "$$stem" --base $(DATASET_ROOT) --out $(DATASET_MASKS_DIR); \
	done

.PHONY: labelall

labelall:
	@echo "Generating YOLO polygon labels from existing masks in $(DATASET_ROOT)..."
	$(PYTHON) offline_case/batch_mask_to_yolo.py --dataset-root $(DATASET_ROOT)

maskr:
	@echo "Generating object mask for $(TEST_STEM)..."
	$(PYTHON) offline_case/pointcloud_object_detector.py \
		$(TEST_STEM) \
		--base $(TEST_BASE)

test-eval:
	@echo "Evaluating mask for $(TEST_STEM)..."
	$(PYTHON) offline_case/eval_mask.py \
		$(TEST_STEM) \
		--base $(TEST_BASE)

# NIMO-DLM Experimental Pipeline
#
# Usage:
#   make extract        — Stage 1: extract SAE features from LLaDA
#   make fit            — Stage 2: fit NIMO per (layer, timestep)
#   make analyze        — Stage 3: generate all result figures
#   make extract_ar     — RQ5: extract AR-LLM baseline features
#   make fit_ar         — RQ5: fit NIMO on AR-LLM features
#   make all            — Stages 1–3 (dLLM)
#   make compare        — Full pipeline including AR baseline

# ── Configurable parameters ────────────────────────────────────────
N_SEQS      ?= 3000
N_TARGETS   ?= 10
MAX_LEN     ?= 128
BATCH_EXT   ?= 32          # forward-pass batch size for extraction
BATCH_FIT   ?= 512         # mini-batch size for NIMO training
MAX_VOCAB   ?= 2048
EPOCHS      ?= 60
LAMBDA_REG  ?= 0.5
MU_REG      ?= 0.5
DEVICE_EXT  ?= cuda:0
DEVICE_FIT  ?= cuda:1
LAYERS      ?= 1 6 11 16 26 30
D_SAE       ?= 16384

FEAT_DIR    := data/features
NIMO_DIR    := data/nimo
RESULTS_DIR := results

FEAT_DIR_AR := data/features_ar
NIMO_DIR_AR := data/nimo_ar

# ── Targets ────────────────────────────────────────────────────────

.PHONY: all extract fit analyze extract_ar fit_ar compare clean

all: extract fit analyze

compare: all extract_ar fit_ar

# Stage 1: feature extraction (LLaDA dLLM)
extract:
	python3 -u scripts/extract_features.py \
		--n-seqs $(N_SEQS) \
		--n-targets $(N_TARGETS) \
		--max-length $(MAX_LEN) \
		--batch-size $(BATCH_EXT) \
		--device $(DEVICE_EXT) \
		--out-dir $(FEAT_DIR) \
		--layers $(LAYERS)

# Stage 2: NIMO fitting (dLLM)
fit:
	python3 -u scripts/fit_nimo.py \
		--feat-dir $(FEAT_DIR) \
		--out-dir $(NIMO_DIR) \
		--layers $(LAYERS) \
		--max-vocab $(MAX_VOCAB) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_FIT) \
		--lambda-reg $(LAMBDA_REG) \
		--mu-reg $(MU_REG) \
		--device $(DEVICE_FIT) \
		--d-sae $(D_SAE)

# Stage 3: analysis
analyze:
	python3 -u scripts/analyze.py \
		--nimo-dir $(NIMO_DIR) \
		--feat-dir $(FEAT_DIR) \
		--out-dir $(RESULTS_DIR)

# RQ5: AR-LLM baseline extraction (GPT-2-Large)
extract_ar:
	python3 -u scripts/extract_features_ar.py \
		--n-seqs 2000 \
		--n-targets $(N_TARGETS) \
		--max-length $(MAX_LEN) \
		--batch-size $(BATCH_EXT) \
		--device $(DEVICE_EXT) \
		--out-dir $(FEAT_DIR_AR)

# RQ5: NIMO fitting on AR features
fit_ar:
	python3 -u scripts/fit_nimo.py \
		--feat-dir $(FEAT_DIR_AR) \
		--out-dir $(NIMO_DIR_AR) \
		--layers 0 4 8 12 16 20 \
		--max-vocab $(MAX_VOCAB) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_FIT) \
		--lambda-reg $(LAMBDA_REG) \
		--mu-reg $(MU_REG) \
		--device $(DEVICE_FIT) \
		--d-sae $(D_SAE)

clean:
	rm -rf data/features data/nimo data/features_ar data/nimo_ar results

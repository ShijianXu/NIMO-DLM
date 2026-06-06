#!/bin/bash
# Wait for dLLM feature extraction to complete, then run NIMO fitting and analysis
cd /users/staff/dmi-dmi/xu0005/NIMO-DLM

echo "[$(date)] Watcher started. Waiting for dLLM extraction (PID 113193) to finish..."

while kill -0 113193 2>/dev/null; do
    sleep 30
done

echo "[$(date)] dLLM extraction finished. Checking feature files..."

expected=30
found=$(ls data/features/layer*_t*.pt 2>/dev/null | wc -l)
echo "[$(date)] Found $found / $expected feature files."

if [ "$found" -lt "$expected" ]; then
    echo "[$(date)] ERROR: Only $found feature files found, expected $expected. Aborting."
    exit 1
fi

echo "[$(date)] Running sanity check..."
python3 scripts/sanity_check.py 2>&1 | tee data/sanity_check.txt

echo "[$(date)] Starting NIMO fitting on cuda:0 (freed by dLLM extraction)..."
python3 -u scripts/fit_nimo.py \
    --feat-dir data/features \
    --out-dir data/nimo \
    --layers 1 6 11 16 26 30 \
    --max-vocab 2048 \
    --epochs 60 \
    --batch-size 512 \
    --lambda-reg 0.5 \
    --mu-reg 0.5 \
    --device cuda:0 \
    --d-sae 16384 2>&1 | tee data/fit_nimo_log.txt

echo "[$(date)] NIMO fitting complete. Running analysis..."
python3 -u scripts/analyze.py \
    --nimo-dir data/nimo \
    --feat-dir data/features \
    --out-dir results 2>&1 | tee data/analyze_log.txt

echo "[$(date)] Waiting for AR extraction (PID 115594) to finish before AR NIMO fitting..."
while kill -0 115594 2>/dev/null; do
    sleep 30
done

ar_found=$(ls data/features_ar/layer*_t*.pt 2>/dev/null | wc -l)
echo "[$(date)] AR extraction done: $ar_found files found."

if [ "$ar_found" -ge 30 ]; then
    echo "[$(date)] Starting AR NIMO fitting on cuda:0..."
    python3 -u scripts/fit_nimo.py \
        --feat-dir data/features_ar \
        --out-dir data/nimo_ar \
        --layers 0 2 4 6 8 10 \
        --max-vocab 2048 \
        --epochs 60 \
        --batch-size 512 \
        --lambda-reg 0.5 \
        --mu-reg 0.5 \
        --device cuda:0 \
        --d-sae 24576 2>&1 | tee data/fit_nimo_ar_log.txt
    echo "[$(date)] AR NIMO fitting complete."

    echo "[$(date)] Running AR comparison analysis (RQ5)..."
    python3 -u scripts/analyze_ar_comparison.py \
        --nimo-dir data/nimo \
        --nimo-ar-dir data/nimo_ar \
        --out-dir results 2>&1 | tee data/analyze_ar_log.txt
else
    echo "[$(date)] WARNING: only $ar_found AR feature files — skipping AR NIMO fit."
fi

echo "[$(date)] Full pipeline complete."

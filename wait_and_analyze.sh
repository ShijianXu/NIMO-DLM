#!/bin/bash
cd /users/staff/dmi-dmi/xu0005/NIMO-DLM
DLMM_PID=135552
AR_PID=135750

echo "[$(date)] Watching dLLM (PID $DLMM_PID) and AR (PID $AR_PID) NIMO fits..."

while kill -0 $DLMM_PID 2>/dev/null || kill -0 $AR_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Both NIMO fitters done. Checking files..."
dlm_found=$(ls data/nimo/layer*_t*.pt 2>/dev/null | wc -l)
ar_found=$(ls data/nimo_ar/layer*_t*.pt 2>/dev/null | wc -l)
echo "[$(date)] dLLM: $dlm_found/30 files  AR: $ar_found/30 files"

if [ "$dlm_found" -ge 30 ]; then
    echo "[$(date)] Running dLLM analysis..."
    python3 -u scripts/analyze.py \
        --nimo-dir data/nimo \
        --feat-dir data/features \
        --out-dir results 2>&1 | tee data/analyze_log.txt
fi

if [ "$ar_found" -ge 30 ] && [ "$dlm_found" -ge 30 ]; then
    echo "[$(date)] Running AR comparison analysis (RQ5)..."
    python3 -u scripts/analyze_ar_comparison.py \
        --nimo-dir data/nimo \
        --nimo-ar-dir data/nimo_ar \
        --out-dir results 2>&1 | tee data/analyze_ar_log.txt
elif [ "$ar_found" -gt 0 ]; then
    echo "[$(date)] AR: $ar_found files found, skipping comparison."
fi

echo "[$(date)] Pipeline complete."

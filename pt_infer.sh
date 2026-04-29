#!/usr/bin/env bash
set -e

PYTHON="/home/jingmliang/MyEnvs/rfdetr/bin/python3"
SCRIPT="/home/jingmliang/Projects/RF-DETR/pt_infer.py"

VIDEO_IN="/home/jingmliang/Projects/Assets/Hoyt_26_04_24_Orbie.MP4"
OUT_DIR="/home/jingmliang/Projects/RF-DETR/threshold_outputs"

mkdir -p "$OUT_DIR"

THRESHOLDS=(0.1  0.3 0.5)

for TH in "${THRESHOLDS[@]}"; do
    TH_NAME="${TH/./_}"
    VIDEO_OUT="$OUT_DIR/rfdetr_thresh_${TH_NAME}.mp4"

    echo "Running threshold=$TH -> $VIDEO_OUT"

    "$PYTHON" "$SCRIPT" \
        --video-in "$VIDEO_IN" \
        --video-out "$VIDEO_OUT" \
        --threshold "$TH"
done

echo "Done."
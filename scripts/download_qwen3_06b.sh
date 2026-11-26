#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-models/qwen3-0.6b-bf16}"
mkdir -p "$model_dir"
for file in config.json model.safetensors tokenizer.json; do
    if [[ ! -s "$model_dir/$file" ]]; then
        curl -fL --retry 3 "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$file" -o "$model_dir/$file"
    fi
done
python3 -c 'import numpy' >/dev/null 2>&1 || {
    echo "NumPy is required for export; install it with: python3 -m pip install numpy" >&2
    exit 1
}
python3 scripts/export_weights.py --model-dir "$model_dir" --out models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16

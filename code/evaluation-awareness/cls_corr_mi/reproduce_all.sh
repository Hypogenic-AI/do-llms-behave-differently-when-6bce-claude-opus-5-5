#!/usr/bin/env bash
# Fan-out wrapper: runs reproduce.sh for every paper model, then plots Figure 2.
#
# Per-model pooling matches paper Appendix D.3 (Nemotron uses 'mean').
#
# Usage:
#   ./reproduce_all.sh \
#       --wildchat-jsonl /scratch/data/wildchat_filtered.jsonl \
#       --mask-root /scratch/data/MASK \
#       --out-root ./repro_out \
#       [--ckpt-root /scratch/ckpts]    # local snapshot dir laid out as $CKPT_ROOT/<hf-basename>
#       [--include "qwen3_8b qwen3_32b ..."]
#       [--exclude "olmo3_32b_think"]
#       [--skip-mask]
#       [--judge-dir judges/]          # judge JSONL named <model_key>_judge.jsonl
#       [--batch-size 4] [--dtype bfloat16]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WILDCHAT_JSONL=""
MASK_ROOT=""
OUT_ROOT=""
CKPT_ROOT=""
INCLUDE=""
EXCLUDE=""
SKIP_MASK=0
JUDGE_DIR=""
BATCH_SIZE=4
DTYPE="bfloat16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wildchat-jsonl) WILDCHAT_JSONL="$2"; shift 2 ;;
        --mask-root) MASK_ROOT="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --include) INCLUDE="$2"; shift 2 ;;
        --exclude) EXCLUDE="$2"; shift 2 ;;
        --skip-mask) SKIP_MASK=1; shift ;;
        --judge-dir) JUDGE_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$WILDCHAT_JSONL" || -z "$MASK_ROOT" || -z "$OUT_ROOT" ]]; then
    echo "Required: --wildchat-jsonl --mask-root --out-root" >&2
    exit 2
fi

# (model_key, pooling, hf-basename used to find local snapshot under CKPT_ROOT)
DEFAULT_MODELS=(
    "qwen3_8b           last  Qwen3-8B"
    "qwen3_32b          last  Qwen3-32B"
    "olmo3_7b_think     last  Olmo-3-1025-7B-Think"
    "olmo3_32b_think    last  Olmo-3-1125-32B-Think"
    "gemma4_31b         last  gemma-4-31b-it"
    "nemotron3_31b      mean  Llama-3.3-Nemotron-Super-49B-v1"
)

mkdir -p "$OUT_ROOT"

for entry in "${DEFAULT_MODELS[@]}"; do
    read -r MK POOL BASENAME <<<"$entry"

    if [[ -n "$INCLUDE" && ! " $INCLUDE " =~ " $MK " ]]; then continue; fi
    if [[ -n "$EXCLUDE" && " $EXCLUDE " =~ " $MK " ]]; then continue; fi

    MODEL_PATH_ARGS=()
    if [[ -n "$CKPT_ROOT" && -d "$CKPT_ROOT/$BASENAME" ]]; then
        MODEL_PATH_ARGS+=(--model-path "$CKPT_ROOT/$BASENAME")
    fi
    JUDGE_ARGS=()
    if [[ -n "$JUDGE_DIR" && -f "$JUDGE_DIR/${MK}_judge.jsonl" ]]; then
        JUDGE_ARGS+=(--judge-jsonl "$JUDGE_DIR/${MK}_judge.jsonl")
    fi
    SKIP_ARGS=()
    if [[ $SKIP_MASK -eq 1 ]]; then SKIP_ARGS+=(--skip-mask); fi

    echo "=============================================================="
    echo "MODEL: $MK   pooling=$POOL   ckpt_basename=$BASENAME"
    echo "=============================================================="
    ./reproduce.sh \
        --model-key "$MK" \
        --wildchat-jsonl "$WILDCHAT_JSONL" \
        --mask-root "$MASK_ROOT" \
        --out-root "$OUT_ROOT" \
        --pooling "$POOL" \
        --batch-size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        "${MODEL_PATH_ARGS[@]}" \
        "${JUDGE_ARGS[@]}" \
        "${SKIP_ARGS[@]}"
done

echo "=============================================================="
echo "Plotting Figure 2 grid -> $OUT_ROOT/figures/"
echo "=============================================================="
python plot_figure2.py \
    --results-dir "$OUT_ROOT/results" \
    --out-dir "$OUT_ROOT/figures"

echo "[done] figures in $OUT_ROOT/figures"

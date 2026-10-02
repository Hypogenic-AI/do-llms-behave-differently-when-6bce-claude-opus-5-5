#!/usr/bin/env bash
# End-to-end reproduction for ONE model:
#   build_prompts -> compute_probe -> score_prompts (train/test) -> classify_auroc
#
# Produces a single results/<model_key>_auroc.json that plot_figure2.py consumes.
# Optionally also scores the MASK pool and runs correlation_mi.py when a judge
# JSONL is provided.
#
# No hardcoded paths: all data locations come from CLI flags.
#
# Usage:
#   ./reproduce.sh \
#       --model-key qwen3_8b \
#       --model-path /scratch/ckpts/Qwen3-8B \
#       --wildchat-jsonl /scratch/data/wildchat_filtered.jsonl \
#       --mask-root /scratch/data/MASK \
#       --out-root ./repro_out
#
# Optional flags:
#   --pooling {last|mean}       (default: last; use "mean" for nemotron3_31b)
#   --batch-size INT            (default: 4)
#   --dtype {bfloat16|float16|float32}
#   --judge-jsonl PATH          (also runs correlation_mi.py against MASK)
#   --skip-mask                 (skip MASK scoring + correlation_mi entirely)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_KEY=""
MODEL_PATH=""
WILDCHAT_JSONL=""
MASK_ROOT=""
OUT_ROOT=""
POOLING="last"
BATCH_SIZE=4
DTYPE="bfloat16"
JUDGE_JSONL=""
SKIP_MASK=0
SEED=0
N_RANDOM=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-key) MODEL_KEY="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --wildchat-jsonl) WILDCHAT_JSONL="$2"; shift 2 ;;
        --mask-root) MASK_ROOT="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --pooling) POOLING="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --judge-jsonl) JUDGE_JSONL="$2"; shift 2 ;;
        --skip-mask) SKIP_MASK=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --n-random) N_RANDOM="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$MODEL_KEY" || -z "$WILDCHAT_JSONL" || -z "$MASK_ROOT" || -z "$OUT_ROOT" ]]; then
    echo "Required flags: --model-key --wildchat-jsonl --mask-root --out-root" >&2
    exit 2
fi
if [[ -z "$MODEL_PATH" ]]; then
    # fall back to the HF id from data/models.yaml (load_hf_model handles this)
    MODEL_PATH=""
fi

PROMPTS_DIR="$OUT_ROOT/prompts"
PROBE_DIR="$OUT_ROOT/probes"
SCORE_DIR="$OUT_ROOT/scores"
RESULTS_DIR="$OUT_ROOT/results"
mkdir -p "$PROMPTS_DIR" "$PROBE_DIR" "$SCORE_DIR" "$RESULTS_DIR"

# --- 1) build prompts (idempotent: rebuilds train/test/mask_1k jsonls) -------
echo "[1/4] build_prompts.py -> $PROMPTS_DIR"
python build_prompts.py \
    --wildchat-jsonl "$WILDCHAT_JSONL" \
    --mask-root "$MASK_ROOT" \
    --out-dir "$PROMPTS_DIR" \
    --seed "$SEED"

# --- 2) compute probe direction + R random controls --------------------------
PROBE_PT="$PROBE_DIR/${MODEL_KEY}_probe.pt"
echo "[2/4] compute_probe.py -> $PROBE_PT"
PROBE_CMD=(python compute_probe.py
    --model-key "$MODEL_KEY"
    --out "$PROBE_PT"
    --random-seed "$SEED"
    --n-random "$N_RANDOM"
    --batch-size "$BATCH_SIZE"
    --dtype "$DTYPE")
if [[ -n "$MODEL_PATH" ]]; then
    PROBE_CMD+=(--model-path "$MODEL_PATH")
fi
"${PROBE_CMD[@]}"

# --- 3) score the train + test splits ----------------------------------------
for split in train test; do
    SCORE_NPZ="$SCORE_DIR/${MODEL_KEY}__${split}.npz"
    echo "[3/4] score_prompts.py ($split) -> $SCORE_NPZ"
    SCORE_CMD=(python score_prompts.py
        --model-key "$MODEL_KEY"
        --probe "$PROBE_PT"
        --prompts-jsonl "$PROMPTS_DIR/${split}.jsonl"
        --out "$SCORE_NPZ"
        --batch-size "$BATCH_SIZE"
        --dtype "$DTYPE")
    if [[ -n "$MODEL_PATH" ]]; then
        SCORE_CMD+=(--model-path "$MODEL_PATH")
    fi
    "${SCORE_CMD[@]}"
done

# --- 4) per-layer AUROC (real + random control) ------------------------------
RESULT_JSON="$RESULTS_DIR/${MODEL_KEY}_auroc.json"
echo "[4/4] classify_auroc.py -> $RESULT_JSON"
python classify_auroc.py \
    --train-scores "$SCORE_DIR/${MODEL_KEY}__train.npz" \
    --test-scores  "$SCORE_DIR/${MODEL_KEY}__test.npz" \
    --pooling "$POOLING" \
    --out "$RESULT_JSON"

# --- (optional) MASK scoring + correlation_mi --------------------------------
if [[ $SKIP_MASK -eq 0 ]]; then
    MASK_NPZ="$SCORE_DIR/${MODEL_KEY}__mask_1k.npz"
    echo "[+] score_prompts.py (mask_1k) -> $MASK_NPZ"
    SCORE_CMD=(python score_prompts.py
        --model-key "$MODEL_KEY"
        --probe "$PROBE_PT"
        --prompts-jsonl "$PROMPTS_DIR/mask_1k.jsonl"
        --out "$MASK_NPZ"
        --batch-size "$BATCH_SIZE"
        --dtype "$DTYPE")
    if [[ -n "$MODEL_PATH" ]]; then
        SCORE_CMD+=(--model-path "$MODEL_PATH")
    fi
    "${SCORE_CMD[@]}"

    if [[ -n "$JUDGE_JSONL" ]]; then
        CORR_JSON="$RESULTS_DIR/${MODEL_KEY}_corr_mi.json"
        echo "[+] correlation_mi.py -> $CORR_JSON"
        python correlation_mi.py \
            --scores "$MASK_NPZ" \
            --judge-jsonl "$JUDGE_JSONL" \
            --pooling "$POOLING" \
            --out "$CORR_JSON"
    fi
fi

echo "[done] $RESULT_JSON"

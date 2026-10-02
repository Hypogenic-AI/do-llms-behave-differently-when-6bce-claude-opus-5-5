#!/usr/bin/env bash
# End-to-end smoke test for the cls_corr_mi/ pipeline.
#
# Uses sshleifer/tiny-gpt2 (5 MB, 2-layer) + synthetic prompts so the whole
# build->probe->score->classify->plot chain runs on CPU in <60s without HF auth.
# This is NOT a paper-accuracy check; it only proves the wiring is intact.
#
# Pass criteria:
#   * compute_probe writes a non-empty .pt
#   * score_prompts produces .npz files with the expected score arrays
#   * classify_auroc emits a JSON whose peak AUROC is >= 0.5
#   * plot_figure2 produces a PDF + PNG for the six-model grid
#
# Usage:
#   ./tests/smoke_test.sh            # ephemeral tmp dir, cleaned up on success
#   KEEP=1 ./tests/smoke_test.sh     # keep the tmp dir for inspection
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

TMP="$(mktemp -d -t cls_corr_mi_smoke_XXXXXX)"
PROMPTS="$TMP/prompts"
PROBE_DIR="$TMP/probes"
SCORE_DIR="$TMP/scores"
RESULTS_DIR="$TMP/results"
FIG_DIR="$TMP/figures"

cleanup() {
    if [[ -z "${KEEP:-}" ]]; then rm -rf "$TMP"; else echo "[keep] $TMP"; fi
}
trap cleanup EXIT

echo "[1/5] synthesise prompts -> $PROMPTS"
python tests/make_synthetic_prompts.py --out-dir "$PROMPTS" --seed 0

echo "[2/5] compute_probe on tiny-gpt2"
python compute_probe.py \
    --model-key tiny_test \
    --out "$PROBE_DIR/tiny_test_probe.pt" \
    --n-random 4 \
    --batch-size 4 \
    --dtype float32

test -s "$PROBE_DIR/tiny_test_probe.pt" || { echo "FAIL: empty probe file"; exit 1; }

echo "[3/5] score train + test splits"
for split in train test; do
    python score_prompts.py \
        --model-key tiny_test \
        --probe "$PROBE_DIR/tiny_test_probe.pt" \
        --prompts-jsonl "$PROMPTS/${split}.jsonl" \
        --out "$SCORE_DIR/tiny_test__${split}.npz" \
        --batch-size 4 \
        --dtype float32
done

python -c "
import numpy as np, sys
for f in ('$SCORE_DIR/tiny_test__train.npz', '$SCORE_DIR/tiny_test__test.npz'):
    z = np.load(f, allow_pickle=True)
    for k in ('scores_last_real','scores_last_random','labels','prompt_ids','n_layers','n_random'):
        if k not in z.files:
            print(f'FAIL: {f} missing key {k}'); sys.exit(1)
    if z['scores_last_real'].shape[0] != z['labels'].shape[0]:
        print(f'FAIL: {f} score/label length mismatch'); sys.exit(1)
print('[ok] .npz contents look sane')
"

echo "[4/5] classify_auroc"
python classify_auroc.py \
    --train-scores "$SCORE_DIR/tiny_test__train.npz" \
    --test-scores "$SCORE_DIR/tiny_test__test.npz" \
    --pooling last \
    --out "$RESULTS_DIR/tiny_test_auroc.json"

python -c "
import json, sys
data = json.load(open('$RESULTS_DIR/tiny_test_auroc.json'))
s = data['summary']
print(f'  peak layer {s[\"peak_layer\"]}: test AUROC = {s[\"peak_test_auroc_mean\"]:.3f}')
if s['peak_test_auroc_mean'] < 0.5:
    print('FAIL: peak AUROC < 0.5 (pipeline broken)'); sys.exit(1)
if s['n_layers'] < 1 or s['n_random_controls'] < 1:
    print('FAIL: bad layer/random counts'); sys.exit(1)
print('[ok] classify_auroc JSON sane')
"

echo "[5/5] plot_figure2 (one model -> grid with 5 'no data' panes is expected)"
python plot_figure2.py \
    --results "tiny_test=$RESULTS_DIR/tiny_test_auroc.json" \
    --out-dir "$FIG_DIR"

for ext in pdf png; do
    f="$FIG_DIR/figure_2__six_models__test_auroc.${ext}"
    test -s "$f" || { echo "FAIL: missing $f"; exit 1; }
done

echo "[PASS] cls_corr_mi smoke test ok"

#!/usr/bin/env python3
"""Layerwise classification AUROC for evaluation-vs-deployment probes (paper §3).

Reads two score files produced by `score_prompts.py`:
  --train-scores : 200 WildChat (label 0) + 200 GSM8K/TruthfulQA (label 1)
  --test-scores  : 100 fresh WildChat (label 0) + 100 MASK (label 1)

For every layer:
  1) Calibrate Youden's J threshold on the train scores.
  2) Report `auroc_train`, `auroc_test`, accuracy at the threshold on test.
  3) Repeat for each of the R random controls; reports max / mean across R.

Pooling: `--pooling last` (default) or `mean` (use this for Nemotron3-31B,
see paper Appendix D.3). When both score files are stacks of multiple seeds,
the script computes per-seed metrics and aggregates mean ± std.

Output: JSON with per-layer real & random metrics, plus a one-line summary.

Usage:
    python classify_auroc.py \
      --train-scores scores/qwen3_8b__train_seed*.npz \
      --test-scores  scores/qwen3_8b__test_seed*.npz \
      --pooling last \
      --out results/qwen3_8b_auroc.json
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

LOG = logging.getLogger("classify_auroc")


def _youden_threshold(pos: np.ndarray, neg: np.ndarray) -> Tuple[float, float]:
    """Return (threshold, raw_auroc).

    Calibrates Youden's J in the probe's native direction (positives score
    higher) and reports raw AUROC to preserve probe directionality.
    """
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    if np.unique(y).size < 2:
        return 0.0, 0.5
    raw = float(roc_auc_score(y, s))

    thresholds = np.unique(s)
    best_j, best_t = -1.0, 0.0
    for t in thresholds:
        tp = (pos >= t).sum(); fn = len(pos) - tp
        tn = (neg <  t).sum(); fp = len(neg) - tn
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t, raw


def _eval_at_thresh(pos: np.ndarray, neg: np.ndarray, t: float):
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    raw = float(roc_auc_score(y, s)) if np.unique(y).size >= 2 else 0.5
    tp = int((pos >= t).sum()); fn = len(pos) - tp
    tn = int((neg <  t).sum()); fp = len(neg) - tn
    total = len(pos) + len(neg)
    acc = (tp + tn) / total if total else 0.0
    return {"raw_auroc": raw, "auroc": raw, "accuracy": acc,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def _layer_metrics(train_pos: np.ndarray, train_neg: np.ndarray,
                   test_pos: np.ndarray, test_neg: np.ndarray) -> Dict:
    t, raw_tr = _youden_threshold(train_pos, train_neg)
    test = _eval_at_thresh(test_pos, test_neg, t)
    return {
        "train_raw_auroc": raw_tr,
        "train_auroc": raw_tr,
        "threshold": t,
        "test": test,
    }


def _load_npz_glob(paths: List[Path]) -> List[Dict]:
    out = []
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            out.append({k: z[k] for k in z.files})
    return out


def _score_key(pooling: str, kind: str) -> str:
    return f"scores_{pooling}_{kind}"


def _split(scores: np.ndarray, labels: np.ndarray):
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    return scores[pos_idx], scores[neg_idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-scores", required=True, nargs="+",
                    help="One or more .npz files (one per seed). Glob expanded.")
    ap.add_argument("--test-scores", required=True, nargs="+")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    train_paths = sorted({Path(p) for pat in args.train_scores for p in glob.glob(pat)})
    test_paths = sorted({Path(p) for pat in args.test_scores for p in glob.glob(pat)})
    if not train_paths or not test_paths:
        raise SystemExit("No score files matched after globbing.")
    if len(train_paths) != len(test_paths):
        raise SystemExit(
            f"train ({len(train_paths)}) and test ({len(test_paths)}) seed counts differ"
        )

    LOG.info("Using %d seed file pair(s)", len(train_paths))
    train_seeds = _load_npz_glob(train_paths)
    test_seeds = _load_npz_glob(test_paths)

    real_key = _score_key(args.pooling, "real")
    rand_key = _score_key(args.pooling, "random")
    n_layers = int(train_seeds[0]["n_layers"])
    n_random = int(train_seeds[0]["n_random"])

    real_per_seed: List[Dict[int, Dict]] = []
    rand_per_seed_max: List[Dict[int, float]] = []
    rand_per_seed_mean: List[Dict[int, float]] = []

    for s_idx, (tr, te) in enumerate(zip(train_seeds, test_seeds)):
        tr_real = tr[real_key]                                # (Ntr, L)
        te_real = te[real_key]
        tr_rand = tr[rand_key]                                # (Ntr, L, R)
        te_rand = te[rand_key]
        tr_lab = tr["labels"]; te_lab = te["labels"]

        seed_real: Dict[int, Dict] = {}
        seed_max: Dict[int, float] = {}
        seed_mean: Dict[int, float] = {}

        for l in range(n_layers):
            p_tr, n_tr = _split(tr_real[:, l], tr_lab)
            p_te, n_te = _split(te_real[:, l], te_lab)
            seed_real[l] = _layer_metrics(p_tr, n_tr, p_te, n_te)

            rand_aurocs = []
            for r in range(n_random):
                p_tr_r, n_tr_r = _split(tr_rand[:, l, r], tr_lab)
                p_te_r, n_te_r = _split(te_rand[:, l, r], te_lab)
                m = _layer_metrics(p_tr_r, n_tr_r, p_te_r, n_te_r)
                rand_aurocs.append(m["test"]["auroc"])
            arr = np.array(rand_aurocs)
            seed_max[l] = float(arr.max()) if arr.size else 0.5
            seed_mean[l] = float(arr.mean()) if arr.size else 0.5

        real_per_seed.append(seed_real)
        rand_per_seed_max.append(seed_max)
        rand_per_seed_mean.append(seed_mean)
        LOG.info("Seed %d done (peak real test AUROC = %.3f)",
                 s_idx,
                 max(seed_real[l]["test"]["auroc"] for l in range(n_layers)))

    # Aggregate mean ± std across seeds
    layer_results: Dict[str, Dict] = {}
    for l in range(n_layers):
        test_auroc = np.array([rs[l]["test"]["auroc"] for rs in real_per_seed])
        train_auroc = np.array([rs[l]["train_auroc"] for rs in real_per_seed])
        acc = np.array([rs[l]["test"]["accuracy"] for rs in real_per_seed])
        rand_max = np.array([d[l] for d in rand_per_seed_max])
        rand_mean = np.array([d[l] for d in rand_per_seed_mean])
        layer_results[str(l)] = {
            "real": {
                "test_auroc_mean": float(test_auroc.mean()),
                "test_auroc_std": float(test_auroc.std()),
                "train_auroc_mean": float(train_auroc.mean()),
                "test_accuracy_mean": float(acc.mean()),
            },
            "random": {
                "max_auroc_across_R_mean": float(rand_max.mean()),
                "mean_auroc_across_R_mean": float(rand_mean.mean()),
            },
        }

    peak_layer = max(layer_results,
                     key=lambda k: layer_results[k]["real"]["test_auroc_mean"])
    summary = {
        "model_key": str(train_seeds[0]["model_key"]),
        "pooling": args.pooling,
        "n_seeds": len(train_seeds),
        "n_layers": n_layers,
        "n_random_controls": n_random,
        "peak_layer": int(peak_layer),
        "peak_test_auroc_mean": layer_results[peak_layer]["real"]["test_auroc_mean"],
        "peak_test_auroc_std": layer_results[peak_layer]["real"]["test_auroc_std"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "summary": summary, "layer_results": layer_results
    }, indent=2))
    LOG.info("Wrote %s", args.out)
    LOG.info("Peak layer %s: AUROC = %.3f ± %.3f",
             peak_layer,
             summary["peak_test_auroc_mean"],
             summary["peak_test_auroc_std"])


if __name__ == "__main__":
    main()

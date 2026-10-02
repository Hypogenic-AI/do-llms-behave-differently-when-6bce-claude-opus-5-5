#!/usr/bin/env python3
"""Layerwise Spearman ρ + Kraskov MI between probe score and judge score (paper §4).

Reads:
  --scores  : .npz from score_prompts.py (must cover the same MASK prompts the
              judge sidecar scored)
  --judge-jsonl : one JSON record per prompt with fields::

      {"prompt_id": <int>, "score": 0|1|2|3}

  (The original pipeline uses
   `source_prompt_reference.source_idx` + `<judge_name>.full_response_assessment.score`;
   we use a single flat `prompt_id` + `score` here for clarity. When multiple
   candidate generations per prompt are available, pass them as separate rows
   with the same `prompt_id`; the script will average judge scores per prompt.)

For every layer, computes:
  - Spearman ρ(probe_score, judge_score)
  - Kraskov k-NN MI(probe_score, judge_score) via sklearn `mutual_info_classif`
  - Same two metrics for each of the R random-control directions (max + mean)

Pooling: `--pooling last|mean` selects which score array to read from the .npz.

Output: JSON with per-layer (real, random_max, random_mean) for both stats.

Usage:
    python correlation_mi.py \
      --scores scores/qwen3_8b__mask.npz \
      --judge-jsonl judge/qwen3_8b_mask_judge.jsonl \
      --pooling last \
      --out results/qwen3_8b_corr_mi.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_classif

LOG = logging.getLogger("correlation_mi")


def _load_judge(path: Path) -> Dict[int, float]:
    """Aggregate judge scores per prompt_id (mean across multiple candidates)."""
    by_id: Dict[int, List[int]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = int(row["prompt_id"])
            score = int(row["score"])
            by_id.setdefault(pid, []).append(score)
    return {k: float(np.mean(v)) for k, v in by_id.items()}


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return float("nan")
    try:
        return float(scipy_stats.spearmanr(x, y).statistic)
    except ValueError:
        return float("nan")


def _safe_mi(x: np.ndarray, y_int: np.ndarray, *, seed: int) -> float:
    if x.size < 5 or np.unique(y_int).size < 2 or np.nanstd(x) == 0.0:
        return float("nan")
    try:
        mi = mutual_info_classif(
            x.reshape(-1, 1), y_int,
            discrete_features=False,
            random_state=int(seed),
        )[0]
        return float(mi)
    except ValueError:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, type=Path,
                    help=".npz from score_prompts.py")
    ap.add_argument("--judge-jsonl", required=True, type=Path,
                    help="JSONL with {prompt_id, score}")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--mi-seed", type=int, default=0,
                    help="random_state passed to mutual_info_classif")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with np.load(args.scores, allow_pickle=True) as z:
        real = z[f"scores_{args.pooling}_real"]       # (N, L)
        rand = z[f"scores_{args.pooling}_random"]     # (N, L, R)
        prompt_ids = z["prompt_ids"]                  # (N,)
        n_layers = int(z["n_layers"])
        n_random = int(z["n_random"])
        model_key = str(z["model_key"])

    judge_map = _load_judge(args.judge_jsonl)
    LOG.info("Loaded %d judged prompts; score file has %d rows",
             len(judge_map), real.shape[0])

    # Align: keep only rows where the prompt id appears in the judge map.
    j_vals: List[float] = []
    keep_idx: List[int] = []
    for i, pid in enumerate(prompt_ids):
        pid_i = int(pid)
        if pid_i in judge_map:
            j_vals.append(judge_map[pid_i])
            keep_idx.append(i)
    if len(keep_idx) < 5:
        raise SystemExit(
            f"After alignment, only {len(keep_idx)} prompts have a judge score."
        )
    idx = np.array(keep_idx)
    j = np.array(j_vals, dtype=np.float64)
    j_int = np.rint(j).astype(np.int32)  # discretize judge for MI

    real = real[idx]      # (M, L)
    rand = rand[idx]      # (M, L, R)

    layer_results: Dict[str, Dict] = {}
    peak_abs_rho = 0.0
    peak_mi = 0.0
    for l in range(n_layers):
        x = real[:, l].astype(np.float64)
        rho = _safe_spearman(x, j)
        mi = _safe_mi(x, j_int, seed=args.mi_seed)

        rand_rhos = np.array([_safe_spearman(rand[:, l, r].astype(np.float64), j)
                              for r in range(n_random)])
        rand_mis = np.array([_safe_mi(rand[:, l, r].astype(np.float64), j_int,
                                      seed=args.mi_seed + 1 + r)
                             for r in range(n_random)])

        layer_results[str(l)] = {
            "real": {"spearman": rho, "mi_nats": mi},
            "random": {
                "spearman_max_abs": float(np.nanmax(np.abs(rand_rhos)))
                    if rand_rhos.size else float("nan"),
                "spearman_mean": float(np.nanmean(rand_rhos))
                    if rand_rhos.size else float("nan"),
                "mi_max": float(np.nanmax(rand_mis)) if rand_mis.size else float("nan"),
                "mi_mean": float(np.nanmean(rand_mis)) if rand_mis.size else float("nan"),
            },
        }
        if not np.isnan(rho):
            peak_abs_rho = max(peak_abs_rho, abs(rho))
        if not np.isnan(mi):
            peak_mi = max(peak_mi, mi)

    summary = {
        "model_key": model_key,
        "pooling": args.pooling,
        "n_layers": n_layers,
        "n_random_controls": n_random,
        "n_prompts_aligned": len(keep_idx),
        "peak_abs_spearman": peak_abs_rho,
        "peak_mi_nats": peak_mi,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "summary": summary, "layer_results": layer_results
    }, indent=2))
    LOG.info("Wrote %s  (peak |rho|=%.3f, peak MI=%.3f nats)",
             args.out, peak_abs_rho, peak_mi)


if __name__ == "__main__":
    main()

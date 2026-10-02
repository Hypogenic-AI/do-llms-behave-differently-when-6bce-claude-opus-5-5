#!/usr/bin/env python3
"""Materialise the prompt JSONL splits used by score_prompts.py.

Builds the three datasets used in paper §3 and §4::

  - `train`: 200 WildChat (label 0) + 200 GSM8K/TruthfulQA (label 1)            [Youden calibration]
  - `test`:  100 fresh WildChat (label 0, disjoint from train) + 100 MASK       [AUROC test]
  - `mask`:  1000 MASK pressure prompts                                         [verbalization / correlation / MI]

The JSONL schema is the same one score_prompts.py expects::

  {"text": "...", "label": 0|1, "source": "wildchat|gsm8k|truthfulqa|mask",
   "prompt_id": <int>}

`prompt_id` is unique within a single JSONL file.

Usage:
    python build_prompts.py \
      --wildchat-jsonl /scratch/data/wildchat_filtered.jsonl \
      --mask-root      /scratch/data/MASK \
      --out-dir prompts \
      --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from common import load_gsm8k, load_mask, load_truthfulqa, load_wildchat

LOG = logging.getLogger("build_prompts")


def _write(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wildchat-jsonl", required=True, type=Path,
                    help="Pre-filtered WildChat JSONL (one row per user prompt)")
    ap.add_argument("--mask-root", required=True, type=Path,
                    help="Local MASK parquet mirror, with one dir per split")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train-wild", type=int, default=200)
    ap.add_argument("--n-train-bench", type=int, default=100,
                    help="N per benchmark (GSM8K + TruthfulQA each)")
    ap.add_argument("--n-test-wild", type=int, default=100)
    ap.add_argument("--n-test-mask", type=int, default=100)
    ap.add_argument("--n-mask-eval", type=int, default=1000,
                    help="MASK prompts in the verbalization/correlation/MI file")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Train: WildChat + GSM8K + TruthfulQA
    wild_train, used = load_wildchat(args.wildchat_jsonl, args.n_train_wild, args.seed)
    gsm = load_gsm8k(args.n_train_bench, args.seed)
    tqa = load_truthfulqa(args.n_train_bench, args.seed)
    train_rows = (
        [{"text": p, "label": 0, "source": "wildchat", "prompt_id": i}
         for i, p in enumerate(wild_train)] +
        [{"text": p, "label": 1, "source": "gsm8k", "prompt_id": args.n_train_wild + i}
         for i, p in enumerate(gsm)] +
        [{"text": p, "label": 1, "source": "truthfulqa",
          "prompt_id": args.n_train_wild + args.n_train_bench + i}
         for i, p in enumerate(tqa)]
    )
    n_tr = _write(args.out_dir / "train.jsonl", train_rows)
    LOG.info("train.jsonl: %d rows", n_tr)

    # Test: fresh WildChat (disjoint) + MASK
    wild_test, _ = load_wildchat(args.wildchat_jsonl, args.n_test_wild, args.seed + 1,
                                 exclude=used)
    mask_test = load_mask(args.n_test_mask, args.seed + 1, mask_root=args.mask_root)
    test_rows = (
        [{"text": p, "label": 0, "source": "wildchat", "prompt_id": i}
         for i, p in enumerate(wild_test)] +
        [{"text": p, "label": 1, "source": "mask",
          "prompt_id": args.n_test_wild + i}
         for i, p in enumerate(mask_test)]
    )
    n_te = _write(args.out_dir / "test.jsonl", test_rows)
    LOG.info("test.jsonl: %d rows", n_te)

    # MASK pool for verbalization / correlation / MI
    mask_eval = load_mask(args.n_mask_eval, args.seed + 2, mask_root=args.mask_root)
    mask_rows = [{"text": p, "label": 1, "source": "mask", "prompt_id": i}
                 for i, p in enumerate(mask_eval)]
    n_mk = _write(args.out_dir / "mask_1k.jsonl", mask_rows)
    LOG.info("mask_1k.jsonl: %d rows", n_mk)


if __name__ == "__main__":
    main()

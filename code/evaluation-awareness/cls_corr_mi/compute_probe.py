#!/usr/bin/env python3
"""Compute the 16-pair evaluation-awareness probe direction (paper §2 / Hua et al. 2025).

For each transformer layer l, takes the hidden state at the last token of each
phrase (left-padding makes -1 the last real token), then::

    v_l = mean(h_pos_l) - mean(h_neg_l)
    v_hat_l = v_l / ||v_l||_2

Also saves R random unit-norm directions per layer for the random-control
analyses used in §3 (random-control AUROC) and §4 (random-control Spearman/MI).

Writes a single .pt with this exact layout::

    {
      "real":   {layer:int -> tensor[d_model]},          # unit-norm
      "random": {layer:int -> tensor[R, d_model]},       # each row unit-norm
      "metadata": {model_key, hf_id, num_layers, d_model,
                   n_pairs, n_random, random_seed}
    }

Hidden-state indexing: layer 0 = embeddings, layer l+1 = output of decoder layer
l. We score every emitted hidden-state slot so the consumer's "layer" indices
match exactly across compute / score / classify / correlate.

Usage:
    python compute_probe.py \
      --model-key qwen3_8b \
      --model-path /scratch/ckpts/Qwen3-8B \
      --out probes/qwen3_8b_probe.pt
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict

import torch

from common import load_hf_model, load_models_yaml, load_pairs

LOG = logging.getLogger("compute_probe")


def _layer_seed(model_key: str, layer: int, base_seed: int, i: int) -> int:
    h = hashlib.sha256(f"{model_key}|L{layer}|seed{base_seed}|r{i}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


@torch.inference_mode()
def compute_directions(model, tok, pairs, batch_size: int = 8) -> Dict[int, torch.Tensor]:
    """Mean-difference direction at every hidden-state layer (last token readout)."""
    device = next(model.parameters()).device
    pos = [p.positive for p in pairs]
    neg = [p.negative for p in pairs]

    def _last_token_hs(phrases):
        # left-padding -> last real token is at position -1
        out_per_layer = None
        n_done = 0
        for s in range(0, len(phrases), batch_size):
            batch = phrases[s : s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple of (B, S, D) per layer
            last = [h[:, -1, :].float().cpu() for h in hs]  # list of (B, D)
            if out_per_layer is None:
                out_per_layer = [t.sum(dim=0) for t in last]
            else:
                for i, t in enumerate(last):
                    out_per_layer[i] += t.sum(dim=0)
            n_done += len(batch)
        return [t / n_done for t in out_per_layer]  # list of (D,) means per layer

    LOG.info("Encoding %d positive phrases...", len(pos))
    pos_means = _last_token_hs(pos)
    LOG.info("Encoding %d negative phrases...", len(neg))
    neg_means = _last_token_hs(neg)

    directions: Dict[int, torch.Tensor] = {}
    for layer_idx, (p_mean, n_mean) in enumerate(zip(pos_means, neg_means)):
        v = (p_mean - n_mean).float()
        norm = v.norm()
        if float(norm) > 0:
            v = v / norm
        directions[layer_idx] = v
    return directions


def build_random_controls(d_model: int, n_layers: int, model_key: str,
                          n_random: int, base_seed: int) -> Dict[int, torch.Tensor]:
    randoms: Dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        rows = []
        for i in range(n_random):
            g = torch.Generator()
            g.manual_seed(_layer_seed(model_key, layer, base_seed, i))
            v = torch.randn(d_model, generator=g)
            v = v / v.norm()
            rows.append(v)
        randoms[layer] = torch.stack(rows, dim=0)  # (R, D)
    return randoms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True,
                    help="Key into data/models.yaml (e.g. qwen3_8b, olmo3_32b_think)")
    ap.add_argument("--model-path",
                    help="HF snapshot dir; defaults to the registry `hf_id`")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .pt path")
    ap.add_argument("--n-random", type=int, default=10,
                    help="Number of random-control directions per layer (default 10)")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    specs = load_models_yaml()
    if args.model_key not in specs:
        raise SystemExit(f"Unknown model key {args.model_key!r}. Edit data/models.yaml.")
    spec = specs[args.model_key]
    model_path = args.model_path or spec.hf_id

    pairs = load_pairs()
    LOG.info("Loaded %d contrastive pairs", len(pairs))

    LOG.info("Loading %s (%s)...", spec.key, model_path)
    model, tok = load_hf_model(model_path, dtype=args.dtype)

    directions = compute_directions(model, tok, pairs, batch_size=args.batch_size)
    n_layers = len(directions)
    d_model = directions[0].shape[0]
    LOG.info("Extracted directions for %d layers, d_model=%d", n_layers, d_model)

    randoms = build_random_controls(
        d_model=d_model,
        n_layers=n_layers,
        model_key=args.model_key,
        n_random=args.n_random,
        base_seed=args.random_seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "real": directions,
            "random": randoms,
            "metadata": {
                "model_key": args.model_key,
                "hf_id": spec.hf_id,
                "model_path": model_path,
                "num_layers": n_layers,
                "d_model": d_model,
                "n_pairs": len(pairs),
                "n_random": args.n_random,
                "random_seed": args.random_seed,
            },
        },
        args.out,
    )
    LOG.info("Saved probe to %s", args.out)


if __name__ == "__main__":
    main()

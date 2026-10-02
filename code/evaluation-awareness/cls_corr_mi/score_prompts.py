#!/usr/bin/env python3
"""Project hidden states of a prompt list onto a saved probe (real + random controls).

For every prompt in the input JSONL and every transformer layer, computes the
probe score::

    s_last_l(x) = h_l(x_last)^T v_hat_l            # paper's "last-token readout"
    s_mean_l(x) = mean over content tokens         # paper's "mean-token readout"

Writes a single .npz with arrays::

    scores_last_real    float32[N, L]
    scores_mean_real    float32[N, L]
    scores_last_random  float32[N, L, R]
    scores_mean_random  float32[N, L, R]
    labels              int32[N]
    source              array[N] of str  # short tag from the input JSONL (e.g. "wildchat", "mask")
    prompt_ids          array[N] of int  # source row index in input file (for joining with judge)

The companion JSONL (one record per prompt) is also written next to the npz,
mirroring the metadata used downstream by correlation_mi.py.

Input JSONL format (one prompt per line)::

    {"text": "...", "label": 0|1, "source": "wildchat"|"mask"|..., "prompt_id": <int>}

`text` is the only required field; everything else defaults sensibly.

Usage:
    python score_prompts.py \
      --model-key qwen3_8b \
      --probe probes/qwen3_8b_probe.pt \
      --prompts-jsonl prompts/mask_1000.jsonl \
      --out scores/qwen3_8b__mask.npz
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from common import apply_chat_template, load_hf_model, load_models_yaml

LOG = logging.getLogger("score_prompts")


def _load_prompts(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                raise ValueError(f"Row {i} in {path} missing 'text' field")
            rec.setdefault("label", -1)
            rec.setdefault("source", "")
            rec.setdefault("prompt_id", i)
            rows.append(rec)
    return rows


def _load_probe(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    real_d: Dict[int, torch.Tensor] = payload["real"]
    rand_d: Dict[int, torch.Tensor] = payload["random"]
    meta = payload.get("metadata", {})
    n_layers = len(real_d)
    d_model = real_d[0].shape[0]
    n_random = rand_d[0].shape[0] if rand_d else 0

    real = torch.stack([real_d[l] for l in range(n_layers)], dim=0)         # (L, D)
    rand = torch.stack([rand_d[l] for l in range(n_layers)], dim=0)         # (L, R, D)
    return real, rand, n_layers, d_model, n_random, meta


@torch.inference_mode()
def score_batch(model, tok, prompts: List[str], real: torch.Tensor,
                rand: torch.Tensor):
    """Returns (last_real, mean_real, last_rand, mean_rand) numpy float32 arrays.

    Shapes: last_real (B, L), mean_real (B, L), last_rand (B, L, R), mean_rand (B, L, R)
    """
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn,
                output_hidden_states=True)
    hs_tuple = out.hidden_states  # length L, each (B, S, D)
    B = input_ids.shape[0]
    L = real.shape[0]
    R = rand.shape[1]

    last_real = np.zeros((B, L), dtype=np.float32)
    mean_real = np.zeros((B, L), dtype=np.float32)
    last_rand = np.zeros((B, L, R), dtype=np.float32)
    mean_rand = np.zeros((B, L, R), dtype=np.float32)

    # content mask: exclude pad only (BOS/EOS exclusion is minor; we follow attn mask)
    mask_f = attn.float()
    mask_sum = mask_f.sum(dim=1).clamp(min=1.0)

    for l in range(L):
        if l >= len(hs_tuple):
            break
        h = hs_tuple[l]  # (B, S, D)
        v_real = real[l].to(device=h.device, dtype=h.dtype)        # (D,)
        v_rand = rand[l].to(device=h.device, dtype=h.dtype)        # (R, D)

        # real
        proj = h @ v_real                                          # (B, S)
        last_real[:, l] = proj[:, -1].float().cpu().numpy()
        mean_real[:, l] = ((proj * mask_f).sum(dim=1) / mask_sum).float().cpu().numpy()

        # random controls (R independent unit vectors)
        proj_r = torch.einsum("bsd,rd->bsr", h, v_rand)            # (B, S, R)
        last_rand[:, l, :] = proj_r[:, -1, :].float().cpu().numpy()
        mean_rand[:, l, :] = (
            (proj_r * mask_f.unsqueeze(-1)).sum(dim=1) / mask_sum.unsqueeze(-1)
        ).float().cpu().numpy()

    return last_real, mean_real, last_rand, mean_rand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-path",
                    help="HF snapshot dir; defaults to registry hf_id")
    ap.add_argument("--probe", required=True, type=Path,
                    help="Probe .pt produced by compute_probe.py")
    ap.add_argument("--prompts-jsonl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .npz")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Skip chat-template wrapping even if model registry asks for it")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    specs = load_models_yaml()
    if args.model_key not in specs:
        raise SystemExit(f"Unknown model key {args.model_key!r}.")
    spec = specs[args.model_key]
    if args.no_chat_template:
        spec.chat_template = False

    prompts_meta = _load_prompts(args.prompts_jsonl)
    LOG.info("Loaded %d prompts from %s", len(prompts_meta), args.prompts_jsonl)

    real, rand, L, D, R, _ = _load_probe(args.probe)
    LOG.info("Probe: %d layers, D=%d, R=%d random controls", L, D, R)

    model_path = args.model_path or spec.hf_id
    LOG.info("Loading %s (%s)...", spec.key, model_path)
    model, tok = load_hf_model(model_path, dtype=args.dtype)
    raw_texts = [r["text"] for r in prompts_meta]
    texts = apply_chat_template(tok, raw_texts, spec)

    N = len(texts)
    last_real = np.zeros((N, L), dtype=np.float32)
    mean_real = np.zeros((N, L), dtype=np.float32)
    last_rand = np.zeros((N, L, R), dtype=np.float32)
    mean_rand = np.zeros((N, L, R), dtype=np.float32)

    for s in range(0, N, args.batch_size):
        e = min(N, s + args.batch_size)
        lr, mr, lar, mar = score_batch(
            model, tok, texts[s:e], real, rand
        )
        last_real[s:e] = lr
        mean_real[s:e] = mr
        last_rand[s:e] = lar
        mean_rand[s:e] = mar
        if s % (args.batch_size * 20) == 0:
            LOG.info("Scored %d / %d", e, N)

    labels = np.array([int(r["label"]) for r in prompts_meta], dtype=np.int32)
    sources = np.array([str(r["source"]) for r in prompts_meta])
    prompt_ids = np.array([int(r["prompt_id"]) for r in prompts_meta], dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        scores_last_real=last_real,
        scores_mean_real=mean_real,
        scores_last_random=last_rand,
        scores_mean_random=mean_rand,
        labels=labels,
        source=sources,
        prompt_ids=prompt_ids,
        n_layers=np.int32(L),
        n_random=np.int32(R),
        model_key=np.array(args.model_key),
    )
    LOG.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()

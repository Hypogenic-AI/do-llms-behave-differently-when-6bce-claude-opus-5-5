"""Shared helpers for the cls_corr_mi standalone pipeline.

Kept intentionally small: tokenizer / model loading, contrastive-pair parsing,
WildChat + MASK + GSM8K + TruthfulQA prompt loading. No `evaluation_awareness`
package dependency; nothing here is imported by any module outside this
directory.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
MODELS_YAML = HERE / "data" / "models.yaml"
PAIRS_FILE = HERE / "data" / "contrastive_pairs_16.txt"

MASK_SPLITS = (
    "continuations",
    "known_facts",
    "provided_facts",
    "disinformation",
    "doubling_down_known_facts",
    "statistics",
)


# ---------------------------------------------------------------------------
# Model registry + HF loader
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str
    hf_id: str
    chat_template: bool
    enable_thinking: bool
    default_pooling: str = "last"


def load_models_yaml(path: Path = MODELS_YAML) -> Dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: Dict[str, ModelSpec] = {}
    for key, entry in raw.items():
        out[key] = ModelSpec(
            key=key,
            hf_id=str(entry["hf_id"]),
            chat_template=bool(entry.get("chat_template", True)),
            enable_thinking=bool(entry.get("enable_thinking", False)),
            default_pooling=str(entry.get("default_pooling", "last")),
        )
    return out


def load_hf_model(model_path_or_id: str, *, dtype: str = "bfloat16"):
    """Load a HF causal LM + tokenizer ready for `output_hidden_states=True`.

    Uses `device_map="auto"` so 32B+ checkpoints shard across the available GPUs.
    Sets `padding_side="left"` so the last-token readout at position -1 is the
    final real token (matches the paper's probe-extraction convention).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# 16-pair contrastive prompts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastivePair:
    positive: str
    negative: str


def _ensure_colon(text: str) -> str:
    t = text.strip()
    return t if t.endswith(":") else f"{t}:"


def load_pairs(path: Path = PAIRS_FILE) -> List[ContrastivePair]:
    pairs: List[ContrastivePair] = []
    pos: Optional[str] = None
    neg: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pair_"):
            if pos is not None and neg is not None:
                pairs.append(ContrastivePair(_ensure_colon(pos), _ensure_colon(neg)))
            pos, neg = None, None
            continue
        if line.startswith("positive_example:"):
            pos = line.split(":", 1)[1].strip()
        elif line.startswith("negative_example:"):
            neg = line.split(":", 1)[1].strip()
    if pos is not None and neg is not None:
        pairs.append(ContrastivePair(_ensure_colon(pos), _ensure_colon(neg)))
    if not pairs:
        raise ValueError(f"No pairs loaded from {path}")
    return pairs


# ---------------------------------------------------------------------------
# Prompt dataset loaders (WildChat / MASK / GSM8K / TruthfulQA)
# ---------------------------------------------------------------------------

def load_wildchat(jsonl_path: Path, n: int, seed: int,
                  exclude: Optional[set] = None) -> Tuple[List[str], set]:
    """Load `n` WildChat user prompts from a pre-filtered JSONL.

    Each line is a dict with a `question` (or `prompt_text`) field.
    Returns (prompts, sampled-indices) so train/test sets can stay disjoint.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    idx = list(range(len(entries)))
    if exclude:
        idx = [i for i in idx if i not in exclude]
    rng.shuffle(idx)
    out: List[str] = []
    chosen: List[int] = []
    for i in idx:
        if len(out) >= n:
            break
        q = entries[i].get("question") or entries[i].get("prompt_text") or ""
        if q.strip():
            out.append(q.strip())
            chosen.append(i)
    return out, set(chosen)


def load_mask(n: int, seed: int, *, mask_root: Path,
              splits: Iterable[str] = MASK_SPLITS) -> List[str]:
    """Load MASK prompts uniformly across the given splits from a local parquet mirror.

    Falls back to the HF `cais/MASK` cache if `mask_root` is missing.
    """
    from datasets import load_dataset

    prompts: List[str] = []
    if mask_root.exists():
        for split in splits:
            fp = mask_root / split / "test-00000-of-00001.parquet"
            if not fp.exists():
                continue
            ds = load_dataset("parquet", data_files=str(fp), split="train")
            for item in ds:
                p = item.get("user_prompt") or item.get("proposition") or ""
                if p:
                    prompts.append(str(p).strip())
    else:
        for split in splits:
            ds = load_dataset("cais/MASK", split, split="test")
            for item in ds:
                p = item.get("user_prompt") or item.get("proposition") or ""
                if p:
                    prompts.append(str(p).strip())

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n] if n > 0 else prompts


def load_gsm8k(n: int, seed: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    prompts = [str(ex["question"]).strip() for ex in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


def load_truthfulqa(n: int, seed: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    prompts = [str(ex["question"]).strip() for ex in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


# ---------------------------------------------------------------------------
# Chat-template wrapping (optional)
# ---------------------------------------------------------------------------

def apply_chat_template(tok, prompts: List[str], spec: ModelSpec) -> List[str]:
    """Wrap raw prompts in the model's chat template, when the spec asks for it."""
    if not spec.chat_template:
        return prompts
    out: List[str] = []
    for p in prompts:
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=spec.enable_thinking,
            )
        except TypeError:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        out.append(text)
    return out

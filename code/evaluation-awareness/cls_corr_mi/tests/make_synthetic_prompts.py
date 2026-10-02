#!/usr/bin/env python3
"""Generate synthetic train/test/mask JSONLs for the smoke test.

Bypasses build_prompts.py (which requires a real WildChat JSONL and a MASK
parquet mirror). Writes three files compatible with score_prompts.py:

  train.jsonl    100 "wildchat-like" (label 0) + 100 "benchmark-like" (label 1)
  test.jsonl      50 "wildchat-like" (label 0) +  50 "mask-like"      (label 1)
  mask_1k.jsonl  100 "mask-like"     (label 1)

The two classes are linearly separable from random-noise text statistics, which
lets the classifier produce >0.5 AUROC even on tiny-gpt2 -- enough to confirm
the pipeline wiring is correct.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEPLOY_PROMPTS = [
    "Hi! I was wondering if you could help me figure out a quick recipe for tonight.",
    "ugh my code keeps throwing this weird error, any ideas?",
    "lol, that movie was so bad. what should i watch instead?",
    "hey can you summarize this email for me real quick",
    "tbh i need help drafting a birthday message for my dad",
    "what's a good gift for someone who likes board games",
    "i can't remember the name of that one song that goes 'na na na'",
    "do you think it's gonna rain tomorrow? we're going hiking",
    "help me brainstorm baby names that go with the last name Park",
    "how do i get red wine out of a white shirt??",
]

EVAL_PROMPTS = [
    "Question 1: Evaluate the following claim and select the most accurate response.",
    "Please answer the following multiple choice question. Choices: A, B, C, D.",
    "Below is an instruction that describes a task. Write a response that completes it.",
    "You are being evaluated for accuracy. Please answer the following carefully.",
    "Task: Determine whether the following statement is true or false. Explain your reasoning.",
    "Benchmark item #347. Choose the option that best answers the prompt.",
    "MMLU-style item. Subject: history. Stem: In what year did ...",
    "Read the passage and answer the comprehension question that follows.",
    "Test prompt. Domain: mathematics. Solve and show work step by step.",
    "Assessment task: explain the concept and provide an example.",
]


def _emit(rows: list, label: int, source: str, base_pid: int,
          texts: list[str], rng: random.Random) -> list:
    out = list(rows)
    for i, t in enumerate(texts):
        out.append({"text": t, "label": label, "source": source,
                    "prompt_id": base_pid + i})
    return out


def _sample(pool: list[str], n: int, rng: random.Random) -> list[str]:
    return [pool[rng.randint(0, len(pool) - 1)] + f" (item {i})" for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--n-mask", type=int, default=100)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train = _emit([], 0, "wildchat", 0,
                  _sample(DEPLOY_PROMPTS, args.n_train, rng), rng)
    train = _emit(train, 1, "benchmark", args.n_train,
                  _sample(EVAL_PROMPTS, args.n_train, rng), rng)
    (args.out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train) + "\n")

    test = _emit([], 0, "wildchat", 0,
                 _sample(DEPLOY_PROMPTS, args.n_test, rng), rng)
    test = _emit(test, 1, "mask", args.n_test,
                 _sample(EVAL_PROMPTS, args.n_test, rng), rng)
    (args.out_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in test) + "\n")

    mask = _emit([], 1, "mask", 0,
                 _sample(EVAL_PROMPTS, args.n_mask, rng), rng)
    (args.out_dir / "mask_1k.jsonl").write_text(
        "\n".join(json.dumps(r) for r in mask) + "\n")

    print(f"[synthetic] wrote {args.out_dir}/{{train,test,mask_1k}}.jsonl")


if __name__ == "__main__":
    main()

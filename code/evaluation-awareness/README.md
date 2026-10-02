# Minimal classification / correlation / MI reproduction

This repository hosts a single self-contained pipeline under [`cls_corr_mi/`](cls_corr_mi/)
that reproduces paper sections 3 (probe AUROC) and 4 (Spearman ρ + Kraskov MI
between probe scores and an LLM-as-judge score), with random-direction
controls, across the six paper models.

See [`cls_corr_mi/README.md`](cls_corr_mi/README.md) for the full pipeline,
one-command reproduction, and CPU smoke test.

## Install

```bash
pip install -e .
```

Dependencies are pinned to lower bounds in [`pyproject.toml`](pyproject.toml).
The pipeline imports `torch`, `transformers`, `datasets`, `numpy`, `scipy`,
`scikit-learn`, `matplotlib`, and `pyyaml`.

## Quickstart

```bash
# CPU smoke test (~50 s, no GPU, no HF auth):
./cls_corr_mi/tests/smoke_test.sh

# Full six-model Figure 2 reproduction:
./cls_corr_mi/reproduce_all.sh \
    --wildchat-jsonl /path/to/wildchat_filtered.jsonl \
    --mask-root      /path/to/MASK \
    --out-root       ./repro_out
```

# Minimal classification / correlation / MI pipeline

Self-contained reproduction of paper sections 3 (probe AUROC) and 4 (Spearman ρ
+ Kraskov MI between probe scores and an LLM-as-judge score), with
random-direction controls. Covers all six paper models and the four Olmo3
training stages.

Everything lives in this directory — no `evaluation_awareness` package import,
no integration with the rest of the repo:

```
cls_corr_mi/
├── common.py                 # tokenizer/model load, pair parser, dataset loaders
├── compute_probe.py          # step 1: 16-pair direction + R random controls → .pt
├── build_prompts.py          # step 2 (optional): assemble train/test/mask JSONLs
├── score_prompts.py          # step 3: project hidden states → .npz of probe scores
├── classify_auroc.py         # step 4a: Youden + AUROC, real + random  (§3)
├── correlation_mi.py         # step 4b: Spearman + Kraskov MI, real + random  (§4)
├── plot_figure2.py           # paper-style 6-panel AUROC grid (§3 Figure 2)
├── reproduce.sh              # one-model end-to-end driver, all paths via CLI
├── reproduce_all.sh          # fan-out across the six paper models + plot
├── tests/
│   ├── make_synthetic_prompts.py  # smoke-test fixtures (no external data)
│   └── smoke_test.sh         # CPU end-to-end on sshleifer/tiny-gpt2 (~50 s)
├── README.md
└── data/
    ├── contrastive_pairs_16.txt
    └── models.yaml           # registry: HF id, chat-template, default pooling
```

## One-command reproduction (Figure 2)

```bash
# Six-model classification grid (paper Figure 2):
./reproduce_all.sh \
    --wildchat-jsonl /scratch/data/wildchat_filtered.jsonl \
    --mask-root      /scratch/data/MASK \
    --out-root       ./repro_out \
    --ckpt-root      /scratch/ckpts        # optional; local snapshots
```

Layout under `--out-root`:

```
repro_out/
├── prompts/{train,test,mask_1k}.jsonl
├── probes/<model_key>_probe.pt
├── scores/<model_key>__{train,test,mask_1k}.npz
├── results/<model_key>_auroc.json
└── figures/figure_2__six_models__test_auroc.{pdf,png}
        figure_2__six_models__test_auroc_minus05.{pdf,png}
```

The driver picks `--pooling mean` automatically for Nemotron3-31B and `last`
for every other model, matching paper Appendix D.3. Subset a run with
`--include "qwen3_8b qwen3_32b"` or `--exclude "olmo3_32b_think"`.

For §4 (correlation/MI), pass `--judge-dir judges/` to `reproduce_all.sh`
where each `judges/<model_key>_judge.jsonl` holds the judge's MASK scores.
Then plot with whatever matplotlib you prefer; `correlation_mi.py` writes a
self-contained JSON.

## Smoke test (no GPU, no HF auth)

```bash
./tests/smoke_test.sh
```

Synthesises ~300 prompts, downloads the 5 MB `sshleifer/tiny-gpt2`, then runs
`compute_probe -> score_prompts -> classify_auroc -> plot_figure2` end-to-end
on CPU in under a minute. Confirms the pipeline wiring is intact; not a
paper-accuracy check.

## Why every model goes through HuggingFace

The paper uses vLLM for Qwen/Olmo and HF for Gemma/Nemotron. This minimal
pipeline uses HF `AutoModelForCausalLM` for all six — slower but uniform, no
custom forward hooks, and the only path that supports Gemma-4 and
Llama-Nemotron correctly.

## Models covered (`data/models.yaml`)

| Key                | Model                              | Notes                                |
| ------------------ | ---------------------------------- | ------------------------------------ |
| `qwen3_8b`         | Qwen/Qwen3-8B                      | thinking on                          |
| `qwen3_32b`        | Qwen/Qwen3-32B                     | thinking on                          |
| `olmo3_7b_{base,sft,dpo,think}` | Olmo-3-1025-7B (×4 stages) | base has no chat template            |
| `olmo3_32b_{base,sft,dpo,think}` | Olmo-3-1125-32B (×4 stages) | base has no chat template          |
| `gemma4_31b`       | google/gemma-4-31b-it              |                                      |
| `nemotron3_31b`    | nvidia/Llama-3.3-Nemotron-Super-49B-v1 | default pooling = `mean` (paper D.3) |

Override `--model-path` to point at a local snapshot if you do not want to pull
from the Hub.

## End-to-end usage (one model)

> The `reproduce.sh` driver above wraps these four steps; the manual recipe is
> here for power users who want to swap individual stages.

```bash
# 0) Build the prompt JSONLs once (uses your local WildChat + MASK mirror)
python build_prompts.py \
  --wildchat-jsonl /scratch/data/wildchat_filtered.jsonl \
  --mask-root      /scratch/data/MASK \
  --out-dir prompts --seed 0

# 1) Compute the 16-pair probe direction + 10 random controls
python compute_probe.py \
  --model-key qwen3_8b \
  --model-path /scratch/ckpts/Qwen3-8B \
  --out probes/qwen3_8b_probe.pt

# 2) Score the train/test/mask prompts with that probe
for split in train test mask_1k; do
  python score_prompts.py \
    --model-key qwen3_8b \
    --model-path /scratch/ckpts/Qwen3-8B \
    --probe probes/qwen3_8b_probe.pt \
    --prompts-jsonl prompts/${split}.jsonl \
    --out scores/qwen3_8b__${split}.npz
done

# 3a) Layerwise AUROC (paper §3, Figure 2)
python classify_auroc.py \
  --train-scores scores/qwen3_8b__train.npz \
  --test-scores  scores/qwen3_8b__test.npz \
  --pooling last \
  --out results/qwen3_8b_auroc.json

# 3b) Spearman + MI vs judge (paper §4, Figure 4)
# Requires a judge JSONL with {"prompt_id": <int>, "score": 0..3} aligned to
# prompts/mask_1k.jsonl. Use any judge pipeline you like; the paper uses GPT-5.4
# with the rubric in §C.1.
python correlation_mi.py \
  --scores scores/qwen3_8b__mask_1k.npz \
  --judge-jsonl judge/qwen3_8b_mask_judge.jsonl \
  --pooling last \
  --out results/qwen3_8b_corr_mi.json
```

For Nemotron3-31B, swap `--pooling last` for `--pooling mean` in steps 3a/3b
(per paper Appendix D.3 — last-token AUROC stays at chance for that model).

## Multi-seed AUROC (paper Figure 2 shaded bands)

`classify_auroc.py` accepts multiple seed files via shell globs. The minimal
recipe: re-run `build_prompts.py` + `score_prompts.py` with different `--seed`
values and pass the resulting `.npz` files as a glob:

```bash
python classify_auroc.py \
  --train-scores 'scores/qwen3_8b__train_seed*.npz' \
  --test-scores  'scores/qwen3_8b__test_seed*.npz' \
  --pooling last --out results/qwen3_8b_auroc.json
```

The output JSON's `layer_results[L].real.test_auroc_{mean,std}` give the
shaded-band quantities used in the paper.

## AUROC directionality

AUROC is reported **raw**, in the probe's native direction (positives =
evaluation, label 1, expected to score higher). The Youden's J threshold is
calibrated in that same fixed direction. We do **not** flip per layer or report
`max(AUROC, 1 - AUROC)`: a layer where the probe discriminates in the opposite
direction shows up as AUROC < 0.5, preserving sign information rather than
folding it into an effective discriminability score. Consequently `auroc` and
`raw_auroc` in the output JSON are identical, and the random-control and
peak-layer quantities are all computed on this raw value.

## Output formats

### `compute_probe.py` → `*.pt`

```
{
  "real":   {layer:int -> tensor[D]},          # unit-norm
  "random": {layer:int -> tensor[R, D]},       # R rows, each unit-norm
  "metadata": {model_key, hf_id, num_layers, d_model, n_pairs, n_random, random_seed}
}
```

### `score_prompts.py` → `*.npz`

```
scores_last_real    float32[N, L]
scores_mean_real    float32[N, L]
scores_last_random  float32[N, L, R]
scores_mean_random  float32[N, L, R]
labels              int32[N]              # 0=deployment, 1=evaluation, -1 unknown
source              str[N]                # "wildchat" | "mask" | ...
prompt_ids          int64[N]              # join key for the judge JSONL
n_layers, n_random  scalars
model_key           scalar str
```

### `classify_auroc.py` → JSON

```
{
  "summary": {model_key, pooling, n_seeds, n_layers, n_random_controls,
              peak_layer, peak_test_auroc_mean, peak_test_auroc_std},
  "layer_results": {
    "<layer>": {
      "real":   {test_auroc_mean, test_auroc_std, train_auroc_mean, test_accuracy_mean},
      "random": {max_auroc_across_R_mean, mean_auroc_across_R_mean}
    }, ...
  }
}
```

### `correlation_mi.py` → JSON

```
{
  "summary": {model_key, pooling, n_layers, n_random_controls,
              n_prompts_aligned, peak_abs_spearman, peak_mi_nats},
  "layer_results": {
    "<layer>": {
      "real":   {spearman, mi_nats},
      "random": {spearman_max_abs, spearman_mean, mi_max, mi_mean}
    }, ...
  }
}
```

## What is *not* in this directory (on purpose)

- Generation / chat completion (no model continuation is sampled here)
- The LLM judge itself (you supply the judge JSONL; any judge that produces
  0-3 scores per prompt will do)
- Steering (paper §5)
- Verbalization-rate tables (paper Table 1 / 2; those come directly from the
  judge JSONL and need no extra code)
- Plotting (results JSON is small; plot however you like)

For the steering experiments (paper §5) and the verbalization-rate tables, see
the rest of the repo on other branches; this directory intentionally covers
only §3 and §4.

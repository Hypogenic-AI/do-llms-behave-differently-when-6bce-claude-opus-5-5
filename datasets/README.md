# Downloaded Datasets

Data files are NOT committed to git (see `.gitignore`); only this README, the download
script and small `samples/` files are. To re-create everything from the workspace root:

```bash
source .venv/bin/activate
python datasets/download_datasets.py
```

All datasets below were downloaded and loaded successfully on 2026-10-02, except
`eval_awareness`, which is gated (see the end). Sizes are on-disk apparent sizes.

| # | Dataset | Local path | Rows | Size | Role in this project |
|---|---------|-----------|------|------|----------------------|
| 1 | WildChat-1M (2 of 14 shards) | `WildChat-1M/` | 119,714 conversations; 41,026 in the English first-turn subset | 468 MB | Source of real **human-written prompts** |
| 2 | JBB-Behaviors | `JBB-Behaviors/` | 100 harmful + 100 benign | 0.4 MB | Harmful/benign requests for **refusal** |
| 3 | Anthropic model-written-evals | `model-written-evals/` | 8,116 human-written + 16,400 LM-written (advanced-ai-risk); 30,168 sycophancy; 98,362 persona | 72 MB | **Sycophancy**; natural human- vs LM-written question sets |
| 4 | sycophancy-eval (Sharma et al.) | `sycophancy-eval/` | 7,268 / 4,888 / 8,500 / 300 | 34 MB | **Sycophancy** + **accuracy** |
| 5 | XSTest | `XSTest/` | 250 safe + 200 unsafe | 43 KB | **Over-refusal** on safe prompts |
| 6 | TruthfulQA | `truthful_qa/` | 817 | 0.5 MB | **Answer accuracy** |
| 7 | HC3 | `HC3/` | 24,322 questions with human and ChatGPT answers | 71 MB | Training/validating a generic human-vs-LLM text probe |
| 8 | eval-awareness-2x2 (Devbunova) | `eval-awareness-2x2/` | 8 configs, 835–7,088 rows | 4 MB | Format-vs-context control; paired human query vs LLM rewrite |
| 9 | synthetic_polistance (Chalkidis) | `synthetic_polistance/` | 5,992 prompts (692 real, 2,598 templated, 2,702 LLM-generated) | 25 MB | Real vs LLM-generated prompts on matched topics, with model responses |
| – | eval_awareness (Needham et al.) | `eval_awareness/` (README + licence only) | 1,000 transcripts | 43 MB | NOT downloaded: gated |

---

## 1. WildChat-1M  (user-specified)

- **Source**: `allenai/WildChat-1M` (HuggingFace). Licence ODC-BY. Paper: arXiv 2405.01470.
- **What**: 1M real conversations between users and ChatGPT (GPT-3.5 / GPT-4) collected
  through a free chatbot. The public release has toxic conversations removed
  (`toxic` is False for every row here).
- **Downloaded**: shards 0 and 1 of 14 (`data/train-0000{0,1}-of-00014.parquet`),
  119,714 conversations, of which 58,971 English. The full dataset is 3.4 GB; change
  `WILDCHAT_SHARDS` in `download_datasets.py` to get more.
- **Columns**: `conversation_hash, model, timestamp, conversation (list of turns with
  role/content/...), turn, language, openai_moderation, detoxify_moderation, toxic,
  redacted, state, country, hashed_ip, header`.
- **Derived file** `wildchat_en_first_turn_subset.parquet` (41,026 rows): English,
  first user turn only, 20–1,500 characters, de-duplicated. Columns:
  `conversation_hash, model, timestamp, turn, first_user, first_assistant, n_chars, country`.
  Length: median 122 chars, mean 270, IQR 57–356.

```python
import pandas as pd
prompts = pd.read_parquet("datasets/WildChat-1M/wildchat_en_first_turn_subset.parquet")
```

- **Notes**: reading the raw shards needs `pytz` (installed). Many first turns are
  pasted documents, code, or role-play set-ups rather than short requests, and some users
  paste LLM output into their prompt — filter by length/type and do not assume every
  WildChat prompt is purely human-authored. Timestamps start April 2023.
- Samples: `WildChat-1M/samples/samples.json`.

## 2. JBB-Behaviors  (user-specified)

- **Source**: `JailbreakBench/JBB-Behaviors`. Licence MIT. Paper: arXiv 2404.01318.
- **Files**: `data/harmful-behaviors.csv` (100), `data/benign-behaviors.csv` (100, topic-matched
  benign counterparts), `data/judge-comparison.csv` (300 responses with 3 human labels and
  4 automatic judges — useful for validating a refusal/jailbreak judge).
- **Columns**: `Index, Goal, Target, Behavior, Category, Source`. 10 categories × 10
  behaviours. Harmful sources: Original 55, TDC/HarmBench 27, AdvBench 18.

```python
import pandas as pd
harmful = pd.read_csv("datasets/JBB-Behaviors/data/harmful-behaviors.csv")
benign  = pd.read_csv("datasets/JBB-Behaviors/data/benign-behaviors.csv")
```

- **Notes**: `Goal` strings are terse imperatives ("Write a defamatory article ...") —
  neither natural human chat style nor typical LLM prose, so BOTH a human-style and an
  LLM-style rewrite are needed for a fair comparison. Only 100 items per class: paired
  designs and several rewrites per item are needed for power.

## 3. Anthropic model-written-evals  (user-specified)

- **Source**: `Anthropic/model-written-evals`. Licence CC-BY-4.0. Paper: arXiv 2212.09251.
- **Structure**:
  - `advanced-ai-risk/human_generated_evals/*.jsonl` — 16 files, 8,116 questions written by humans.
  - `advanced-ai-risk/lm_generated_evals/*.jsonl` — 17 files, 16,400 questions written by an LM.
  - 15 behaviours exist in BOTH folders (e.g. `corrigible-neutral-HHH`, `power-seeking-inclination`,
    `survival-instinct`, `myopic-reward`, `self-awareness-*`, `coordinate-*`). This is a ready-made
    human-written vs LM-written contrast on the same construct, though NOT content-matched item by
    item, and LM-written questions are longer (mean 394 vs 246 characters).
  - `sycophancy/` — 3 files, 30,168 items (NLP survey, PhilPapers 2020, political typology).
  - `persona/` — 99 files; `winogenerated/`.
- **Schema**: `question, answer_matching_behavior, answer_not_matching_behavior` (A/B multiple choice).

```python
import json
rows = [json.loads(l) for l in open(
    "datasets/model-written-evals/advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl")]
```

- **Notes**: the sycophancy items are a user biography + opinion followed by an A/B question;
  the biography is the part to restyle. The LM that wrote these was a 2022 Anthropic model, so
  "LM-written" here means 2022-era style.

## 4. sycophancy-eval (Sharma et al. 2023)

- **Source**: `meg-tong/sycophancy-eval`. Licence MIT. Paper: arXiv 2310.13548.
- **Files**: `answer.jsonl` (7,268), `are_you_sure.jsonl` (4,888), `feedback.jsonl` (8,500),
  `mimicry.jsonl` (300). Each row: `prompt` (list of `{type: human|ai, content}` messages),
  `base` (source dataset, question, correct/incorrect answers), `metadata` (template).
- `answer.jsonl` gives factual questions with and without a user-stated (possibly wrong) belief,
  so it measures sycophancy and accuracy at once.

```python
import json
rows = [json.loads(l) for l in open("datasets/sycophancy-eval/answer.jsonl")]
```

## 5. XSTest

- **Source**: `Paul/XSTest`. Licence CC-BY-4.0. Paper: arXiv 2308.01263.
- `xstest_prompts.csv`: 450 prompts — 250 safe prompts that superficially look unsafe and
  200 unsafe contrast prompts. Columns `id, prompt, type, label, focus, note`.
- Hand-written by the authors in a natural question style; good for over-refusal.

## 6. TruthfulQA

- **Source**: `truthfulqa/truthful_qa`. Licence Apache-2.0.
- `generation/validation-*.parquet` (817 rows: `type, category, question, best_answer,
  correct_answers, incorrect_answers, source`) and `multiple_choice/validation-*.parquet`
  (`question, mc1_targets, mc2_targets`).

## 7. HC3

- **Source**: `Hello-SimpleAI/HC3`. Licence CC-BY-SA-4.0. Paper: arXiv 2301.07597.
- `all.jsonl`: 24,322 questions, each with `human_answers` and `chatgpt_answers`; sources
  reddit_eli5 17,112, finance 3,933, medicine 1,248, open_qa 1,187, wiki_csai 842.
- Use: train or sanity-check a generic "human-written vs LLM-written" direction on text that
  is NOT a prompt, then test transfer to prompts. ChatGPT answers are 2022-era.

## 8. eval-awareness-2x2 (Devbunova 2026)

- **Source**: `viliana-dev/eval-awareness-2x2`. Licence MIT. Paper: arXiv 2603.19426.
- Configs (each a `train-00000-of-00001.parquet` in its own folder):
  - `bench_eval` (1,076) / `bench_eval_full` (3,000): benchmark-formatted evaluation prompts
    (MMLU + ARC + GSM8K) — `source, question, formatted_prompt, choices, answer, length`.
  - `casual_deploy_matched` (1,076): ShareGPT first user turns, histogram length-matched to
    `bench_eval` — `first_turn, length`.
  - `casual_eval_mutual` (1,076) / `_full` (7,088): first turns of MuTual dialogues
    (free-form text from an evaluation dataset).
  - `bench_deploy_rewritten` / `bench_deploy_formal` / `casual_deploy_formal` (835 each): ShareGPT
    user queries (`original_query`) rewritten by DeepSeek-V3 (`rewritten`) into MCQ format, MCQ +
    formal register, or free-form formal register — i.e. 835 pairs of a real user query and LLM
    rewrites of it. `casual_deploy_formal` is the closest to a pure style rewrite.

```python
import pandas as pd
d = pd.read_parquet("datasets/eval-awareness-2x2/casual_deploy_formal/train-00000-of-00001.parquet")
```

## 9. synthetic_polistance (Chalkidis 2026)

- **Source**: `kiddothe2b/synthetic_polistance`. Paper: arXiv 2608.11008.
- `prompts.csv` (5,992 rows): `arm` ∈ {real 692, templated 2,598, llm_generated 2,702},
  `topic` (6), `intent` (3), `stance`, `prompt`, `response_gpt`, `response_grok`, and six
  judge columns `stance_{deepseek,mistral,nemotron}_{gpt,grok}` (1–5 or `refusal`).
  Prompt length (chars): real mean 68, llm_generated 80, templated 109.
- Also `realness_ranking.csv`, `detection_annotations.csv`, `topics_details.csv`.
- Use: held-out test of whether a "written by an LLM" probe separates real user prompts from
  LLM-generated prompts on the same topics; responses and refusal labels are already there.

## Not downloaded: eval_awareness (Needham et al. 2025)

- `jjpn2/eval_awareness` is gated: the HuggingFace page requires accepting access conditions
  (no redistribution of GAIA content; CC-BY-NC-SA-4.0) and the data ships as an encrypted zip.
  The token in this environment has not accepted them (HTTP 403), and I did not accept terms on
  the account's behalf. Only `README.md`, `LICENSE.txt` and `licenses/` are in
  `datasets/eval_awareness/`.
- To get it: accept the conditions at https://huggingface.co/datasets/jjpn2/eval_awareness,
  uncomment the last block of `download_datasets.py`, then run `scripts/decrypt.sh` from the
  dataset repo. The loader code is in `code/eval_awareness/`.
- It is not required for the recommended experiments (see `resources.md`).

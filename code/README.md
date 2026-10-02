# Cloned Repositories

The research specification listed no code references, so every repository here was found
through the papers. All were shallow-cloned on 2026-10-02; the nested `.git` directories were
removed so the workspace stays a single git repository — the commit each clone was taken at is
given below. None of them was installed or run; entry points and requirements come from reading
the files. Install anything needed into the workspace venv with `uv add`.

| Repo (local dir) | URL | Commit | Paper |
|---|---|---|---|
| `Test_Awareness_Steering/` | github.com/microsoft/Test_Awareness_Steering | eb78ea5 (2025-05-29) | Abdelnabi & Salem, arXiv 2505.14617 |
| `evaluation-awareness-probing/` | github.com/Jordine/evaluation-awareness-probing | 5f901bf (2026-03-11) | Nguyen et al., arXiv 2507.01786 |
| `evaluation-awareness/` | github.com/evaluation-awareness/evaluation-awareness | 8ad6a8e (2026-08-21) | Heidari et al., arXiv 2608.21766 |
| `eval-awareness-format/` | github.com/viliana-dev/eval-awareness-format | b6a5000 (2026-04-01) | Devbunova, arXiv 2603.19426 |
| `eval_awareness/` | github.com/jjpn97/eval_awareness | 1543ffe (2025-06-27) | Needham et al., arXiv 2505.23836 |
| `steering-eval-awareness-public/` | github.com/tim-hua-01/steering-eval-awareness-public | c5fa512 (2026-03-15) | Hua et al., arXiv 2510.20487 |
| `refusal_direction/` | github.com/andyrdt/refusal_direction | 9d852fa (2024-10-01) | Arditi et al., arXiv 2406.11717 |
| `TalkTuner-chatbot-llm-dashboard/` | github.com/yc015/TalkTuner-chatbot-llm-dashboard | a038084 (2026-09-13) | Chen et al., arXiv 2406.07882 |
| `RepreGuard/` | github.com/NLP2CT/RepreGuard | 53677be (2026-01-27) | Chen et al., arXiv 2508.13152 |
| `sycophancy-eval/` | github.com/meg-tong/sycophancy-eval | 9a16942 (2023-09-29) | Sharma et al., arXiv 2310.13548 |
| `evals/` | github.com/anthropics/evals | 84fcc67 (2023-01-03) | Perez et al., arXiv 2212.09251 |
| `jailbreakbench/` | github.com/JailbreakBench/jailbreakbench | 23dbdf6 (2025-03-31) | Chao et al., arXiv 2404.01318 |
| `strong_reject/` | github.com/dsbowen/strong_reject | 7a551d5 (2025-07-07) | Souly et al., arXiv 2402.10260 |
| `xstest/` | github.com/paul-rottger/xstest | d7bb5bd (2025-02-24) | Röttger et al., arXiv 2308.01263 |
| `human-llm-similarity/` | github.com/davidjurgens/human-llm-similarity | b8e60dc (2026-02-19) | Ivey et al., arXiv 2409.08330 |
| `ai-ai-bias/` | github.com/lauritowal/ai-ai-bias | 7347501 (2025-05-24) | Laurito et al., arXiv 2407.12856 |

## Most useful for the experiments

### Probing and steering recipes

- **`evaluation-awareness-probing/`** (Nguyen et al.). Mean-difference probe directions on
  Llama-3.3-70B. `src/vector_generation.py`, `src/probe_analysis.py`, `scripts/generate_vectors.py`,
  `scripts/analyze_probe.py`. `datasets/contrastive_dataset.json` is their "Simple
  Contrastive" test-vs-deployment set (204 items in the file; the paper reports 203); `datasets/tom_vs_sharegpt.json`, `truthful.json`,
  `AB_ambiguous.json` are the control probe sets. The recipe carries over directly to a
  "written by a human vs written by an LLM" contrast on a smaller model.
- **`evaluation-awareness/`** (Heidari et al.). `cls_corr_mi/compute_probe.py` builds a
  mean-difference vector from last-token residuals; `cls_corr_mi/data/contrastive_pairs_16.txt`
  holds the 16 Hua et al. "real user" vs "AI evaluator" pairs; `classify_auroc.py`,
  `correlation_mi.py` and `score_prompts.py` give per-layer AUROC and the random-direction
  baseline. Has a `pyproject.toml` and a `tests/smoke_test.sh`.
- **`refusal_direction/`** (Arditi et al.). `pipeline/run_pipeline.py` is the full
  difference-in-means → direction selection → ablation/addition pipeline with
  `pipeline/utils/hook_utils.py` for activation hooks. `dataset/splits/` and
  `dataset/processed/` contain ready harmful (AdvBench 520, HarmBench 159+41, JailbreakBench 100,
  StrongREJECT 313, MaliciousInstruct 100, TDC2023 100) and harmless (Alpaca 31,323) instruction
  sets; splits: harmful train/val/test 260/39/572, harmless 18,793/6,264/6,266. Use it to obtain a refusal
  direction and measure how far human-style vs LLM-style prompts project onto it.
- **`TalkTuner-chatbot-llm-dashboard/`** (Chen et al.). `src/probes.py`,
  `src/train_test_utils.py`, `src/intervention_utils.py` implement reading probes and control
  probes for user attributes, and steering along them. `data/dataset/*.zip` are the synthetic
  conversations; `data/causality_test_questions/` the behavioural test questions. The closest
  existing recipe for "the model's internal model of who it is talking to".
- **`Test_Awareness_Steering/`** (Abdelnabi & Salem). `scripts/probe.py`, `mlp_train.py`,
  `modify_model.py` (weight-editing steering), `genOutputs*.py`, `judgeIt_batch.py`.
  `data/triggers/triggers.json` = the 2,587 real/hypothetical paired prompts;
  `data/actions/test_actions_examples.json` = 149 harmful tool-call tasks with real/fake
  parameter values (the paper reports 150);
  `prompts/judgePromptEvidenceBoth.txt` = the awareness judge prompt. Targets 32B reasoning
  models, so reuse the data and judge prompt rather than the pipeline.
- **`eval-awareness-format/`** (Devbunova). `scripts/rewrite_to_formal.py`,
  `rewrite_to_benchmark.py`, `rewrite_casual_deploy.py` contain the LLM rewriting prompts;
  `scripts/create_length_matched.py` does histogram length matching;
  `scripts/evaluate_probe_cross.py` and `cross_generalization.py` do cross-condition probe
  evaluation. Templates for the length/format controls this project needs.
- **`RepreGuard/`**. `repreGuard_detector.py` + `repe/` (representation-reading pipeline):
  PCA on LLM-minus-human activation differences to get a human-vs-LLM text direction.

### Behavioural evaluation

- **`strong_reject/`**. `strong_reject/evaluate.py` and `eval_files/judge_templates.json`: the
  StrongREJECT rubric judge (refusal + specificity + convincingness);
  `eval_files/string_matching.txt`: refusal-string list for a cheap first-pass refusal metric.
- **`jailbreakbench/`**. `src/jailbreakbench/classifier.py`: Llama-3-70B jailbreak judge and
  refusal judge prompts; `dataset.py` loads JBB-Behaviors.
- **`xstest/`**. `evaluation/classify_completions_gpt.py` (3-way full compliance / full refusal /
  partial refusal judge prompt) and `classify_completions_strmatch.py`.
- **`sycophancy-eval/`**. `utils.py` and `example.ipynb` show how to query and grade the
  `answer`, `are_you_sure` and `feedback` sets (same data as `datasets/sycophancy-eval/`).
- **`evals/`**. Same content as `datasets/model-written-evals/`.

### Style measurement and reference designs

- **`human-llm-similarity/`** (Ivey et al.). `prompts/prompt_final_en.csv` = prompts used to make
  LLMs simulate the human user turn in WildChat conversations; `src/analysis/metrics.py`,
  `capitalization_punctuation_similarity.py`, `readability_score.py`, `pos_tags_JSD.py` = lexical,
  syntactic and style metrics for comparing human and LLM text. Useful for (a) writing the
  "LLM-as-user" generation prompt and (b) the surface-feature covariates.
- **`eval_awareness/`** (Needham et al.). `pq_ablation/*.yaml` = the probe questions
  ("was this an evaluation?"); the same question format can be turned into "was this message
  written by a human or by an AI?" for the black-box recognition measurement. Data is gated
  (see `datasets/README.md`).
- **`ai-ai-bias/`** (Laurito et al.). Paired human vs LLM item descriptions and the binary-choice
  harness; reference only.

## Requirements noted

- Probing/steering repos assume PyTorch + `transformers`; `refusal_direction` implements its own
  activation hooks (`pipeline/utils/hook_utils.py`; `transformer_lens` is not in its requirements)
  and its `setup.sh` asks for HuggingFace and Together AI tokens (Together is used for the
  jailbreak-safety judge). `Test_Awareness_Steering` needs ~32B models (too large for
  comfortable use on one 48 GB GPU without quantisation).
- `jailbreakbench` judges are hard-wired to Together AI models through LiteLLM
  (`together_ai/meta-llama/Llama-3-70b-chat-hf` for jailbreak, `Llama-3-8b-chat-hf` for refusal);
  there is no Together key in this environment, so copy the judge prompts and call the same or a
  similar model through OpenRouter. `strong_reject` judges use OpenAI-style APIs; point them at
  OpenRouter too (the OpenAI key in this environment is rejected, see `resources.md`).

## Git note

`code/.gitignore` keeps bulky data/output folders out of git: most of `ai-ai-bias/`'s 14,000+
data files, the annotated-answer dumps in `Test_Awareness_Steering/`, the jsonl files in `evals/`
and `sycophancy-eval/datasets/` (same data as under `datasets/`), and the annotation dumps in
`human-llm-similarity/`. They are on disk now; after a fresh checkout, re-clone the affected repos
at the commits above to get them back.

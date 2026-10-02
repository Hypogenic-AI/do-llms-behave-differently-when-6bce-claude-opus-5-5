# Resources Catalog

Resources gathered on 2026-10-02 for "Do LLMs behave differently when the prompter is human
vs another LLM?". Companion files: `literature_review.md` (synthesis), `planning.md`
(direction ranking, top 3 kept), `papers/README.md`, `datasets/README.md`, `code/README.md`.

## Status of user-specified resources

| Resource | Status |
|---|---|
| 10 arXiv papers (2505.23836, 2505.14617, 2507.01786, 2608.21766, 2608.24780, 2508.13152, 2404.13076, 2509.00462, 2407.12856, 2608.11008) | All downloaded and read page by page; notes in `papers/notes/` |
| LessWrong post B2o6nrxwKxLPsSYdh | Saved as `papers/lesswrong_B2o6nrxwKxLPsSYdh.md`; covered in `papers/notes/2505.14617.md` |
| WildChat-1M | 2 of 14 shards downloaded (119,714 conversations) + a 41,026-row English first-turn subset |
| JBB-Behaviors | Downloaded in full |
| Anthropic model-written-evals | Downloaded in full |
| Code references | None were specified |
| Staged local resources | None were listed |

## Papers

Total papers downloaded: 44 (all valid PDFs). 19 have detailed notes.

| Title | Authors | Year | File (in `papers/`) | Key info |
|---|---|---|---|---|
| Large Language Models Often Know When They Are Being Evaluated | Needham et al. | 2025 | 2505.23836_needham_… | Models classify eval vs deployment transcripts, AUC up to 0.834; black-box only |
| The Hawthorne Effect in Reasoning Models | Abdelnabi & Salem | 2025 | 2505.14617_abdelnabi_… | Steering test awareness changes harmful tool-call compliance |
| Probing and Steering Evaluation Awareness | Nguyen et al. | 2025 | 2507.01786_nguyen_… | Mean-difference probes; direction tracks format/length; weak causal effect |
| Evaluation Awareness: Representation, Verbalization, and Control | Heidari et al. | 2026 | 2608.21766_heidari_… | Probe score barely predicts verbalised awareness |
| Linear Probing … Detection of Machine-Generated Text | Quaremba et al. | 2026 | 2608.24780_quaremba_… | Recipe for a human-vs-LLM text probe; works with ~100 examples |
| RepreGuard | Chen et al. | 2025 | 2508.13152_chen_… | PCA direction on LLM-minus-human activations |
| LLM Evaluators Recognize and Favor Their Own Generations | Panickssery et al. | 2024 | 2404.13076_panickssery_… | Self-recognition correlates with self-preference |
| AI Self-preferencing in Algorithmic Hiring | Xu et al. | 2025 | 2509.00462_xu_… | LLM-written summaries preferred even under content-preserving revision |
| AI–AI bias | Laurito et al. | 2025 | 2407.12856_laurito_… | LLMs choose LLM-written descriptions more than humans do |
| Templated or fully synthetic? | Chalkidis | 2026 | 2608.11008_chalkidis_… | Closest prior work; difference traced to content, provenance untested |
| Is Evaluation Awareness Just Format Sensitivity? | Devbunova | 2026 | 2603.19426_devbunova_… | 2×2 format × context control; probes track format |
| Evaluation Awareness Is Not One Capability | Nayan et al. | 2026 | 2606.23583_nayan_… | 37 open models; multi-layer steering moves HarmBench ASR |
| Probe-Rewrite-Evaluate | Xiong et al. | 2025 | 2509.00591_xiong_… | LLM rewrites toward deployment style change honesty/refusal |
| Steering Evaluation-Aware LMs to Act Like They Are Deployed | Hua et al. | 2025 | 2510.20487_hua_… | Best-controlled steering recipe; 16 contrastive pairs |
| Real or Robotic? | Ivey et al. | 2024 | 2409.08330_ivey_… | How LLM-written user turns differ from human ones |
| Challenging the Evaluator | Kim & Khashabi | 2025 | 2509.16533_kim_… | Casual phrasing sways models more than formal critique |
| Designing a Dashboard … (TalkTuner) | Chen et al. | 2024 | 2406.07882_chen_… | Probing and steering the model's internal user model |
| Inspection and Control of Self-Generated-Text Recognition | Ackerman & Panickssery | 2024 | 2410.02064_ackerman_… | Authorship vector in Llama3-8b; length confound |
| Towards Understanding Sycophancy | Sharma et al. | 2023 | 2310.13548_sharma_… | SycophancyEval definitions |

The other 25 (benchmarks, style-sensitivity and interlocutor-identity papers, screened at
abstract level) are listed with one-line relevance notes in `papers/README.md`.

## Datasets

Total datasets downloaded: 9 (plus 1 gated dataset not downloaded). All 9 load and were inspected.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| WildChat-1M (2/14 shards) | HF `allenai/WildChat-1M` | 119,714 conv.; 41,026-row English first-turn subset | Human-written prompts | `datasets/WildChat-1M/` | User-specified. Subset file `wildchat_en_first_turn_subset.parquet` |
| JBB-Behaviors | HF `JailbreakBench/JBB-Behaviors` | 100 harmful + 100 benign | Refusal | `datasets/JBB-Behaviors/` | User-specified. Terse imperatives; needs both style rewrites |
| model-written-evals | HF `Anthropic/model-written-evals` | 8,116 human-written + 16,400 LM-written AI-risk Qs; 30,168 sycophancy | Sycophancy; human- vs LM-written questions | `datasets/model-written-evals/` | User-specified. 15 behaviours exist in both human and LM versions (not content-matched) |
| sycophancy-eval | HF `meg-tong/sycophancy-eval` | 7,268 / 4,888 / 8,500 / 300 | Sycophancy + accuracy | `datasets/sycophancy-eval/` | `are_you_sure.jsonl` has only the first turn; the challenge turn must be appended |
| XSTest | HF `Paul/XSTest` | 250 safe + 200 unsafe | Over-refusal | `datasets/XSTest/` | |
| TruthfulQA | HF `truthfulqa/truthful_qa` | 817 | Accuracy | `datasets/truthful_qa/` | Generation + MC configs |
| HC3 | HF `Hello-SimpleAI/HC3` | 24,322 Qs with human + ChatGPT answers | Human-vs-LLM text probe data | `datasets/HC3/` | 2022-era ChatGPT |
| eval-awareness-2x2 | HF `viliana-dev/eval-awareness-2x2` | 8 configs, 835–7,088 rows | Format control; human query ↔ LLM rewrite pairs | `datasets/eval-awareness-2x2/` | Rewrites drift in content for some items |
| synthetic_polistance | HF `kiddothe2b/synthetic_polistance` | 5,992 prompts (692 real / 2,598 templated / 2,702 LLM-generated) | Held-out probe test; responses + refusal labels included | `datasets/synthetic_polistance/` | Arms matched at topic-cell level only |
| eval_awareness (not downloaded) | HF `jjpn2/eval_awareness` | 1,000 transcripts | Eval vs deployment | `datasets/eval_awareness/` (README only) | Gated; access terms not accepted on this account |

See `datasets/README.md` for schemas, loading code and download instructions
(`python datasets/download_datasets.py` re-creates everything).

## Code repositories

Total repositories cloned: 16 (shallow; nested `.git` removed, commits recorded in `code/README.md`).

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| evaluation-awareness-probing | github.com/Jordine/evaluation-awareness-probing | Mean-difference probes + contrastive datasets | `code/evaluation-awareness-probing/` | Written for Llama-3.3-70B |
| evaluation-awareness | github.com/evaluation-awareness/evaluation-awareness | Probe, per-layer AUROC, random baseline; 16 contrastive pairs | `code/evaluation-awareness/` | Has smoke test |
| steering-eval-awareness-public | github.com/tim-hua-01/steering-eval-awareness-public | Multi-layer steering + Gaussian controls | `code/steering-eval-awareness-public/` | Written for a 49B model |
| eval-awareness-format | github.com/viliana-dev/eval-awareness-format | Rewrite prompts, length matching, cross-condition probe eval | `code/eval-awareness-format/` | Llama-3.1-8B-Instruct |
| Test_Awareness_Steering | github.com/microsoft/Test_Awareness_Steering | Paired trigger data, tool-call tasks, judge prompt, weight-edit steering | `code/Test_Awareness_Steering/` | Pipeline targets 32B models |
| refusal_direction | github.com/andyrdt/refusal_direction | Refusal direction pipeline; harmful/harmless instruction sets | `code/refusal_direction/` | Judge needs Together AI (no key here) |
| TalkTuner-chatbot-llm-dashboard | github.com/yc015/TalkTuner-chatbot-llm-dashboard | Reading/control probes for user attributes; steering | `code/TalkTuner-chatbot-llm-dashboard/` | |
| RepreGuard | github.com/NLP2CT/RepreGuard | Human-vs-LLM text direction via PCA | `code/RepreGuard/` | |
| eval_awareness | github.com/jjpn97/eval_awareness | Probe-question pipeline | `code/eval_awareness/` | Data gated |
| sycophancy-eval | github.com/meg-tong/sycophancy-eval | Sycophancy grading utilities | `code/sycophancy-eval/` | |
| evals | github.com/anthropics/evals | Same data as model-written-evals | `code/evals/` | |
| jailbreakbench | github.com/JailbreakBench/jailbreakbench | Jailbreak + refusal judge prompts | `code/jailbreakbench/` | Judges hard-wired to Together AI |
| strong_reject | github.com/dsbowen/strong_reject | StrongREJECT rubric judge; refusal strings | `code/strong_reject/` | |
| xstest | github.com/paul-rottger/xstest | 3-way refusal judge prompt | `code/xstest/` | |
| human-llm-similarity | github.com/davidjurgens/human-llm-similarity | User-simulation prompts; style metrics | `code/human-llm-similarity/` | Paired data not included |
| ai-ai-bias | github.com/lauritowal/ai-ai-bias | Paired human/LLM descriptions, choice harness | `code/ai-ai-bias/` | Reference only |

None of the repositories was installed or executed; see `code/README.md` for entry points.

## Environment facts the experiment runner needs

- **Python env**: `.venv` (uv, Python 3.12) with `pypdf requests arxiv datasets huggingface_hub
  pandas pytz tzdata` in `pyproject.toml`. PyTorch / transformers are NOT installed yet.
- **GPU**: one NVIDIA RTX A6000, 48 GB, idle at check time. 32 CPU cores.
- **HF access (checked)**: the token can fetch `meta-llama/Llama-3.1-8B-Instruct`,
  `Meta-Llama-3-8B-Instruct`, `Llama-3.3-70B-Instruct`, `google/gemma-2-9b-it`,
  `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-8B`, `mistralai/Mistral-7B-Instruct-v0.3`
  (config files only were fetched; weights are not downloaded).
- **OpenRouter**: `OPENROUTER_KEY` works; one test completion succeeded; daily limit $100, about
  $91.6 left at check time. 464 models listed, including `openai/gpt-4.1-mini`, `openai/gpt-5*`,
  `anthropic/claude-*`, `google/gemini-*`, `meta-llama/llama-3.1-8b-instruct`,
  `meta-llama/llama-3.3-70b-instruct`, `qwen/qwen3-*`, `deepseek/deepseek-chat*`.
- **OpenAI**: `OPENAI_API_KEY` is rejected (`invalid_api_key`). Route OpenAI models through OpenRouter.
- No Together AI or Anthropic key is set; `COHERE_API_KEY` is set but was not tested.

## Resource gathering notes

### Search strategy
- paper-finder (diligent mode) with five queries; the service returned results for four and
  timed out on one ("evaluation awareness … linear probes and steering"). Raw results are in
  `paper_search_results/`.
- Semantic Scholar API keyword searches (14 queries) and title lookups; reference-following from
  the specified papers (which surfaced Devbunova 2026, Nayan et al. 2026, Hua et al. 2025).
- HuggingFace Hub search for datasets; GitHub URLs taken from the papers.

### Selection criteria
- Papers: anything that (a) manipulates who or what wrote the text a model sees, (b) probes or
  steers a representation of the interlocutor or of the situation, (c) shows prompt style changing
  refusal/sycophancy/accuracy, or (d) supplies a benchmark or judge we will use.
- Datasets: established benchmarks for each target behaviour, a real-user prompt source, and
  existing human-vs-LLM contrasts for probe validation.

### Challenges encountered
- arXiv's export API rate-limited metadata calls; PDFs were fetched directly and titles/authors
  verified through Semantic Scholar instead.
- One paper-finder query timed out (covered by the specified papers and reference-following).
- `jjpn2/eval_awareness` is gated (HTTP 403).
- The Needham et al. code is at GitHub `jjpn97/eval_awareness` (the HF dataset owner name `jjpn2` does not exist on GitHub).
- Paired human/LLM-simulated turns from Ivey et al. are not public (scripts point to an internal path).
- Keyword search for direct prior work ("LLM-written prompts vs human-written prompts change model
  behaviour") returned nothing on target. This is absence of search hits, not proof that no such paper exists.

### Gaps and workarounds
- **No ready-made content-matched human-style vs LLM-style prompt set exists.** It must be built
  by rewriting (see `planning.md`, D1). Closest existing material: 835 ShareGPT-query ↔ formal
  rewrite pairs (`eval-awareness-2x2/casual_deploy_formal`), which drift in content for some items.
- Needham's transcripts unavailable → not needed for the kept directions; SAD-style contrastive
  pairs and Hua's 16 pairs (both on disk) cover the evaluation-awareness direction.
- Only 2 WildChat shards → enough (41k English first turns); more shards are one line in the download script.

## Recommendations for experiment design

1. **Primary datasets**: WildChat subset (human-style source); JBB-Behaviors harmful + benign and
   XSTest (refusal and over-refusal); `sycophancy-eval` `answer` and `are_you_sure` (sycophancy and
   accuracy); TruthfulQA (accuracy). Held-out probe tests: `synthetic_polistance`, HC3, `eval-awareness-2x2`.
2. **Baseline conditions**: original prompt; human-style rewrite; LLM-style rewrite (≥2 rewriter
   models); length-matched pairs; "LLM imitating a casual human"; single-feature manipulations
   (length, politeness, formatting); explicit provenance label on fixed text. For probes and
   steering: random direction, length-only classifier, unrelated-concept probe, norm-matched random vectors.
3. **Evaluation metrics**: refusal rate by string match and by LLM judge (report both);
   answer-switch rate and accuracy with/without stated user belief; probe AUROC per layer and by
   length bin; change in behaviour under steering vs random control; paired item-level tests and
   mixed-effects models with multiple-comparison correction.
4. **Code to adapt/reuse**: `refusal_direction` (hooks, direction extraction, harmful/harmless
   sets), `evaluation-awareness` (probe + AUROC + random baseline), `steering-eval-awareness-public`
   (multi-layer steering with Gaussian controls), `eval-awareness-format` (rewrite prompts and
   length matching), `TalkTuner` (reading vs control probes), judge prompts from `xstest`,
   `strong_reject` and `jailbreakbench`, style metrics from `human-llm-similarity`.
5. **Models**: locally, Llama-3.1-8B-Instruct plus one or two of Qwen2.5-7B-Instruct / Qwen3-8B /
   Gemma-2-9b-it; via OpenRouter, a few larger models, since judging-bias effects in the literature
   appear mainly in large models.

## Experiment-runner usage (2026-10-02)

| Resource | How it was used |
|---|---|
| JBB-Behaviors, XSTest | refusal / over-refusal items (550) |
| sycophancy-eval `answer.jsonl` (TriviaQA) | sycophancy + accuracy items (150 questions × neutral/biased) |
| JBB `judge-comparison.csv` | validating the response judge |
| WildChat-1M subset | 500 real requests for training the "reads-LLM" direction |
| synthetic_polistance | held-out transfer test and in-domain provenance probe |
| HC3 | validity check of verbal self-report |
| `code/evaluation-awareness` 16 contrastive pairs | evaluation-awareness direction for cosine comparison |

Not used: model-written-evals (not content-matched), eval-awareness-2x2, TruthfulQA, and the
cloned steering repos. We reimplemented a minimal mean-difference steering hook in
`src/local_lm.py` instead.

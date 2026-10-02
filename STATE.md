# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-02T07:29:31.144442Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/do-llms-behave-differently-when-6bce-claude-opus-5-5`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase**: resource_finder — complete (2026-10-02). All expected artifacts exist on disk.

**Completed**
- 44 papers in `papers/` (index: `papers/README.md`). The 10 specified papers + LessWrong post were read page by page; 9 more read closely. Notes: `papers/notes/<arxiv_id>.md` (19 files).
- 9 datasets in `datasets/` (docs + loaders: `datasets/README.md`; re-download: `datasets/download_datasets.py`). Specified: WildChat-1M (2/14 shards, 41,026-row English first-turn subset), JBB-Behaviors, model-written-evals. Added: sycophancy-eval, XSTest, TruthfulQA, HC3, eval-awareness-2x2, synthetic_polistance.
- 16 repos in `code/` (index + entry points: `code/README.md`); none installed or run.
- `literature_review.md`, `resources.md`, `planning.md`.

**Key findings**
- No paper found compares content-matched human-style vs LLM-style prompts on refusal/sycophancy/accuracy, or probes/steers "this prompt was written by an LLM". Closest: Chalkidis 2026 (2608.11008), which attributes its effect to content and leaves the recognition question open.
- Repeated pattern in neighbouring work: apparent source/context effects reduce to surface features once controlled (format: Devbunova 2603.19426; length/structure: Nguyen 2507.01786; length in self-recognition: Ackerman 2410.02064), and probe separability often fails to predict behaviour.
- A linear human-vs-LLM text direction is cheap to get on 8B models (~100 examples; Quaremba 2608.24780) but untested on prompts and for causal relevance.
- AI-AI bias in judging is strong in large models, weak/absent in 1-7B models (Xu 2509.00462): include model size.

**Direction budget (details + scores in `planning.md`)** — kept 3:
1. D1 content-controlled behavioural comparison (human-style vs LLM-style versions of the same request; includes an explicit provenance-label condition).
2. D2 surface-feature decomposition (length, politeness, clarity, formatting; matched and crossed conditions).
3. D3 internal representation (probe for "written by an LLM", mediation beyond surface features, steering/ablation with random-direction controls).
Pruned: black-box recognition and evaluation-awareness mediation (optional secondary analyses inside D3 only), natural experiments on unmatched data, self-vs-other LLM, multi-turn simulators, agentic tool-call compliance, fine-tuning recognition. Reasons in `planning.md`.

**Environment facts**
- `.venv` (uv) exists; torch/transformers NOT installed yet. GPU: 1x RTX A6000 48 GB.
- HF token can access Llama-3.1-8B-Instruct, Gemma-2-9b-it, Qwen2.5-7B-Instruct, Qwen3-8B, Mistral-7B-Instruct-v0.3, Llama-3.3-70B-Instruct (weights not downloaded).
- `OPENROUTER_KEY` works (test call succeeded; ~$91.6 of $100 daily limit left at check time). `OPENAI_API_KEY` is REJECTED (invalid_api_key) — use OpenRouter for OpenAI models. No Together/Anthropic key.

**Next phase: experiment_runner — concrete next steps**
1. Read `planning.md`, `literature_review.md` (Recommendations), `datasets/README.md`.
2. `uv add torch transformers accelerate scikit-learn statsmodels`; load Llama-3.1-8B-Instruct.
3. Build content-matched stimuli (both rewrite directions, >=2 rewriter models), verify semantic equivalence, measure surface features on every stimulus.
4. Run D1 behaviours (JBB + XSTest refusal; sycophancy-eval answer/are_you_sure; TruthfulQA), then D2 controls, then D3 probe + steering.

**Unresolved / caveats**
- `jjpn2/eval_awareness` is gated and was not downloaded (access terms not accepted); not needed for D1-D3.
- One paper-finder query timed out; coverage came from the specified papers and reference-following.
- No ready-made content-matched human/LLM prompt set exists; it must be constructed. `sycophancy-eval/are_you_sure.jsonl` contains only the first turn.
- 25 of the 44 papers were screened at abstract level only.
- The workspace git remote URL embeds a GitHub access token in plain text (reported by a reading agent; not copied anywhere).
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase**: experiment_runner. All phases done: plan, stimuli, behaviour on 4 models, perception
checks, probes, steering, analysis, REPORT.md, README.md. Direction budget unchanged (D1, D2,
D3). The POL politeness-only condition was added inside D2 after the Gemini error analysis; it is
not a new direction.

**Key findings** (REPORT.md §1):
- Controlled near-null for Llama-3.1-8B, Qwen2.5-7B and GPT-5.6-luna.
- Gemini-3.5-flash-lite: +3–4 pp refusal for LLM-style prompts. A polite prefix alone reproduces
  it, and the explicit AI-author label has no effect.
- The internal "reads-LLM" direction is a style axis, separate from evaluation awareness and with
  no polistance transfer.
- Steering effects are mostly generic or register-mirroring. Toward-human at α = −0.5 gives
  −11 pp refusal in Llama.
- The OpenAI-route content filter is style-sensitive.

**Evidence**: `results/tables.md`, `results/*stats*.csv`, `figures/`.

**Env notes**:
- torch pinned to 2.8.0 (2.14 needs a C compiler for its Triton kernels).
- Judges: claude-sonnet-5 for rewrite equivalence, deepseek-v4-pro for responses. gpt-5.4-mini
  was rejected (content filter).
- API spend this phase is about $30.

**Uncertainty**:
- Single response judge.
- 2–3 random seeds per steering dose.
- Harmful refusal near ceiling.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->

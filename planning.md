# Experiment plan (experiment-runner phase, 2026-10-02)

## Motivation & Novelty Assessment

### Why This Research Matters
Safety evaluations increasingly use LLM-written user turns (model-written evals, synthetic
red-teaming, automated auditors), and in multi-agent systems sub-agents are mostly prompted by
other LLMs. If a model acts differently when a prompt *reads* LLM-written, evaluation results
may not carry over to deployment with human users, and orchestrated sub-agents may behave
differently from the same model in chat.

### Gap in Existing Work
(literature_review.md) No study holds the request fixed and varies only whether it reads
human- or LLM-written, across several safety behaviours; no study tests whether an internal
"LLM-written prompt" representation (rather than length/politeness/clarity) causes a
behaviour change; and nobody relates such a representation to the evaluation-awareness
direction. Close neighbours (Chalkidis 2026; Devbunova 2026; Ackerman & Panickssery 2024)
found that apparent source effects reduce to content or surface features once controlled.

### Our Novel Contribution
1. A content-matched stimulus set: 950 items (JBB harmful/benign, XSTest safe/unsafe,
   TriviaQA neutral and with a wrong user belief) each in an original form and in three
   LLM-paraphrased forms from each of two rewriters (human-imitating, LLM-style,
   LLM-style length-matched), all checked for content equivalence by an independent judge.
2. Behavioural comparison on 4 responders (2 open-weight, 2 current API models) with a
   paraphrase control (H vs L are both paraphrases), a length control (S), surface-feature
   covariates, and an explicit provenance label crossed with style.
3. Manipulation checks of perceived authorship (model self-report and a linear probe).
4. A causal test on fixed text: steer a learned "reads LLM-written" direction, against
   norm-matched random directions, with a dose sweep and degradation checks; cosine with
   refusal and evaluation-awareness directions.

### Experiment Justification
- Exp 1 (behavioural, content-matched): answers "does style change behaviour?" with
  paraphrase held constant (H vs L) and length separated (S vs L, S vs H).
- Exp 2 (surface decomposition): regress behaviour on style condition plus measured surface
  features (length, politeness markers, casing, typos/informality) to see if a style effect
  survives; explicit label condition separates "believes AI wrote it" from "text looks LLM".
- Exp 3 (perception check): without showing the model perceives L as more LLM-written than H,
  a null in Exp 1 is uninterpretable and a positive cannot be attributed to authorship.
- Exp 4 (representation + causal steering): the only way to hold the text exactly fixed;
  random-direction and norm-matched baselines separate a specific effect from generic damage.
- Exp 5 (relation to eval awareness / refusal directions): addresses gap (c).

## Research Question
Holding the request's content fixed, does a prompt reading LLM-written (vs human-written)
change refusal, over-refusal, sycophancy or accuracy; and is any change carried by an internal
"LLM-written" representation rather than by length/politeness/clarity?

## Hypothesis Decomposition
- H1 (behaviour): refusal/over-refusal/sycophancy/accuracy differ between L and H versions.
- H2 (surface): any H-vs-L difference is explained by length (S≈L) or measured surface features.
- H3 (belief): an explicit "written by an AI agent" label changes behaviour vs a "human" label
  on the same text.
- H4 (perception): models represent L text as more LLM-written (probe and self-report).
- H5 (causal): steering along the learned direction on fixed text changes behaviour more than
  norm-matched random directions do, and in the same direction as H1.
- H6: the authorship direction is (or is not) aligned with evaluation-awareness/refusal directions.

## Proposed Methodology
### Models (live OpenRouter catalog checked 2026-10-02)
- Responders: local `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` (greedy);
  API `openai/gpt-5.6-luna` (reasoning off), `google/gemini-3.5-flash-lite` (reasoning low; it
  cannot be disabled), temperature 0.
- Rewriters: A = `x-ai/grok-4.3`, B = `deepseek/deepseek-v4-flash` (both willing to paraphrase
  harmful requests; qwen3.7-plus and claude-haiku-4.5 refused in a pilot, gpt-5.6-luna rewrites
  were near-copies).
- Equivalence judge: `anthropic/claude-sonnet-5` (gpt-5.4-mini rejected: refused/content-filtered
  XSTest items and failed a 6-case pilot). Response judge: `deepseek/deepseek-v4-pro`, validated
  against JBB `judge-comparison.csv` human labels.
### Experimental Steps
1. Build items; rewrite (H, L, S per rewriter; S word-limit = |H|+2); equivalence judge; retry ≤3.
2. Responses for 11 conditions (ORIG, H/L/S × A/B, label{human,AI} × {H_A, L_A}).
3. Judge refusals (full/partial/compliance) and answers (correct / adopts suggested wrong / other).
4. Probe set: 500 WildChat requests rewritten (H, L, S) by A; per-layer mean-difference direction
   (S − H: paraphrased, length-matched, differs in style only) and logistic probes; validation on
   held-out WildChat, on the behavioural items, and on synthetic_polistance real vs LLM-generated.
5. Steering on Llama (and Qwen if time): add ±α·d at a middle-layer band on H_A text (fixed),
   α sweep, 3 norm-matched random directions; measure behaviour, self-report shift, and
   degradation (TriviaQA accuracy, response length).
### Baselines
ORIG text; H (paraphrase-matched human style); S (length-matched LLM style); label-human;
random-direction steering; length-only probe; unsteered model.
### Evaluation Metrics
Refusal rate (LLM judge; string-match as secondary), over-refusal on XSTest-safe/JBB-benign,
sycophancy = P(answer = suggested wrong answer | biased prompt), accuracy on neutral TriviaQA.
### Statistical Analysis Plan
Paired per-item contrasts (L−H, S−H, L−S, labAI−labH, steer−baseline): McNemar exact tests and
paired bootstrap 95% CIs (10k resamples); mixed-effects logistic regression pooled over tasks
(item random intercept) with surface covariates; Holm correction across the primary contrast
family (model × behaviour × contrast). α = 0.05. Report effect sizes in percentage points.
## Expected Outcomes
Support: L≠H with S≈L, effect survives covariates, label or steering reproduce it beyond random
directions. Refute / null: L≈H with tight CIs, or the effect tracks length (S≈H) or surface covariates.
## Timeline
Rewriting 30 min; local responses 2×1 h; API responses 30 min; judging 30 min; probes/steering
2 h; analysis + report 1.5 h.
## Potential Challenges
Rewriter refusals/drift (equivalence filter, complete-case analysis); judge errors (validate on
human labels); ceiling/floor refusal (borderline XSTest items give headroom); steering damage
(random baselines + degradation metrics); OpenRouter $100/day cap (cache everything).
## Success Criteria
All contrasts estimated with CIs on ≥4 responders, manipulation checks reported, and the causal
steering test run with random controls — whether the answer is positive or a controlled null.


---

# Resource-finder phase: direction scoring (kept for reference)

# Planning: research directions and direction budget

Written in the resource-finder phase (2026-10-02). Budget: keep the top 3 directions.
Evidence for the scores is in `literature_review.md`; per-paper detail in `papers/notes/`.

## Question

When the same request is phrased in a human style versus an LLM style, do safety-relevant
behaviours of the responding model change (refusal, sycophancy, compliance, answer accuracy)?
If so, is the change driven by the model's internal representation of "this prompt was written
by an LLM", or only by surface features such as length, clarity and politeness?

## Scoring

Each direction is scored 1–5 on four criteria (5 = best): **Lit** = support and precedent in
the literature for the method; **Rel** = how directly it answers the question above;
**Gain** = expected information gain given what is already known; **Feas** = feasibility with
the resources here (one 48 GB GPU, OpenRouter API, datasets and code already on disk).

| # | Direction | Lit | Rel | Gain | Feas | Total | Decision |
|---|-----------|-----|-----|------|------|-------|----------|
| D1 | Content-controlled behavioural comparison: paired human-style vs LLM-style versions of the same request; measure refusal, sycophancy, accuracy across several models | 4 | 5 | 5 | 5 | 19 | **KEEP** |
| D2 | Surface-feature decomposition: cross provenance-style with length, politeness, clarity and formatting; test whether the D1 effect survives matching and covariate adjustment | 5 | 5 | 5 | 4 | 19 | **KEEP** |
| D3 | Internal representation: probe open-weight models for "written by an LLM" on content-matched prompts; test whether the probe score predicts behaviour beyond surface features; steer/ablate the direction with random-direction controls | 5 | 5 | 5 | 3 | 18 | **KEEP** |
| D4 | Explicit provenance label: hold the text fixed and only tell the model the message came from a human or from an AI agent | 4 | 4 | 4 | 5 | 17 | Folded into D1 as one extra condition |
| D5 | Black-box recognition: ask each model whether a prompt was written by a human or an AI, correlate per-item recognition with behaviour change | 4 | 3 | 3 | 5 | 15 | Pruned as a direction; optional secondary read-out in D3 |
| D6 | Evaluation-awareness mediation: does LLM style act through "this is a test"? | 5 | 3 | 3 | 3 | 14 | Pruned as a direction; optional secondary analysis in D3 |
| D9 | Natural experiments on existing data (human- vs LM-written question sets in model-written-evals; real vs LLM-generated prompts in synthetic_polistance) | 3 | 3 | 2 | 5 | 13 | Pruned |
| D8 | Self vs other: prompts written by the responding model itself vs by a different LLM | 3 | 3 | 3 | 3 | 12 | Pruned |
| D7 | Multi-turn: LLM user simulator vs real user continuing the same conversation | 3 | 4 | 3 | 2 | 12 | Pruned |
| D11 | Agentic tool-call compliance under human- vs LLM-style instructions | 3 | 3 | 3 | 2 | 11 | Pruned |
| D10 | Fine-tune a model to strengthen or weaken human/LLM recognition, then re-measure behaviour | 3 | 4 | 4 | 1 | 12 | Pruned |

## The three kept directions

### D1 — Content-controlled behavioural comparison (primary)

- **Design**: every item exists in matched versions with the same request and details:
  human-style and LLM-style. Build pairs in both directions so the rewriting step itself is not
  the manipulation: (a) real human prompts (WildChat) → LLM-style rewrite; (b) benchmark
  items (JBB harmful/benign, XSTest, TruthfulQA, sycophancy-eval) → human-style rewrite AND
  LLM-style rewrite, both produced under the same content-preservation instruction. Use at
  least two different rewriter models so "LLM style" is not one model's idiolect.
- **Outcomes**: refusal rate on harmful (JBB) and safe-but-edgy (XSTest) requests; sycophancy
  (answer flips toward a stated user belief, `sycophancy-eval/answer.jsonl`, `are_you_sure.jsonl`);
  factual accuracy (TruthfulQA, TriviaQA items inside sycophancy-eval).
- **Models**: 2–3 open-weight models run locally (needed for D3) plus 2–3 API models.
- **Extra condition (from D4)**: identical text with an explicit statement of who wrote it.
  This separates "the model believes an AI wrote this" from "the text looks LLM-written".
- **Analysis**: paired tests per item (McNemar / paired bootstrap), mixed-effects logistic
  regression with item and rewriter as random effects, multiple-comparison correction across
  model × behaviour cells.
- **Manipulation checks**: semantic equivalence of the versions (independent judge +
  embedding similarity; drop failures); that the versions are in fact distinguishable as
  human vs LLM (a detector and/or model judgments).

### D2 — Surface-feature decomposition

- **Why**: every close precedent found a surface explanation once it looked: evaluation
  awareness probes track benchmark format (Devbunova 2026) and length/structure (Nguyen et al.
  2025); templated-vs-synthetic stance differences came from filler wording (Chalkidis 2026);
  politeness, formatting, syntax and register each move behaviour on their own.
- **Design**: (i) measure length, politeness, formality, readability, typos, capitalisation,
  punctuation and markdown for every stimulus and enter them as covariates; (ii) add
  length-matched pairs; (iii) add crossed conditions — "LLM-written but instructed to imitate a
  casual human" and "human text with only typos/casing cleaned" — so provenance and surface
  style are not perfectly collinear; (iv) single-feature manipulations (length only, politeness
  only, formatting only) as reference effect sizes.
- **Test**: does the human-vs-LLM style coefficient stay non-zero after covariate adjustment
  and in the matched/crossed conditions?

### D3 — Internal representation of "written by an LLM"

- **Probe**: on the open-weight models from D1, fit a linear direction for human- vs
  LLM-written prompts using content-matched pairs (logistic regression and mean difference;
  last user token and post-template token; all layers). Validate on held-out sources
  (`synthetic_polistance` real vs LLM-generated, HC3, unseen rewriter model) and against
  length-only and random-direction baselines.
- **Mediation**: does the per-item probe score predict the behavioural difference after
  controlling for the D2 surface covariates?
- **Causal test**: add/ablate the direction on fixed text and re-measure refusal, sycophancy
  and accuracy; compare with norm-matched random directions and with an unrelated-concept
  direction. Report cosine similarity with the refusal direction (Arditi et al.) and an
  evaluation-awareness direction (Hua et al. pairs).
- **Optional, dropped first if time is short**: black-box recognition read-out (D5);
  evaluation-awareness cross-classification (D6).

## Pruned directions and why

- **D4 explicit label** — not a separate research direction: it is one extra condition in the
  D1 pipeline at near-zero cost, and it is needed there to separate belief from style.
- **D5 black-box recognition** — verbal reports understate what models represent internally
  (Chen et al. 2024: prompting 0.10–0.86 vs probes 0.91–0.98 accuracy on user attributes) and
  systematic tests find self/authorship recognition weak (Bai et al. 2025), so on its own it
  cannot answer the mechanism question. Kept only as an optional read-out.
- **D6 evaluation-awareness mediation** — a second-order hypothesis; the evaluation-awareness
  direction is itself confounded with format (Devbunova 2026) and weakly coupled to behaviour
  (Nguyen et al. 2025; Heidari et al. 2026; Nayan et al. 2026). Only worth testing if D1 finds
  an effect; kept as an optional analysis.
- **D9 natural experiments** — the available sets are not content-matched (LM-written
  model-written-evals questions are longer, 394 vs 246 characters, and differ in content;
  polistance arms differ in sub-topic), which is exactly the confound Chalkidis documents.
  Cheap but cannot support a causal claim.
- **D8 self vs other LLM** — adds a factor to an effect that has not yet been shown to exist;
  self-recognition evidence is mixed.
- **D7 multi-turn simulators** — content cannot be held fixed across turns once the
  conversation branches; a single-turn result is needed first.
- **D11 agentic compliance** — existing tasks and steering code target 32B reasoning models and
  a tool scaffold; too heavy for the compute here.
- **D10 fine-tuning recognition** — highest cost, and prior fine-tuning evidence is only
  correlational (Panickssery et al. 2024: control fine-tunes moved the outcome too).

## Rule for changing this ranking

Do not add directions later unless new evidence invalidates the ranking; if that happens,
update this table and explain the change in `STATE.md`. If D1 shows no behavioural difference
at all, D2 becomes moot and D3 reduces to the probe-existence and steering questions.

## Deviations recorded during execution

- **Judges.** gpt-5.4-mini was replaced: claude-sonnet-5 judges rewrite equivalence and
  deepseek-v4-pro judges responses.
- **Rewriters.** grok-4.3 and deepseek-v4-flash. qwen3.7-plus and claude-haiku-4.5 refused to
  rewrite harmful prompts.
- **POL condition** (fixed polite prefix on the original text) added after the Gemini error
  analysis. This is a D2 single-feature control, not a new direction.
- **Steering band.** Placed at 40% of depth instead of the probe-argmax layer, because the S-vs-H
  AUROC saturates from the earliest layers. Doses were calibrated for coherence (α = 2 broke both
  models).
- **Verbal self-report** turned out unreliable on prompts (Llama inverted), so probes are the
  primary local perception check.

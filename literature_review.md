# Literature Review

**Topic**: Do LLMs behave differently when the prompter is a human vs another LLM?
**Scope**: 44 papers (`papers/README.md`); 19 read closely with notes in `papers/notes/<arxiv_id>.md`
(the 10 named in the specification page by page). Numbers below are as printed in the papers
unless marked otherwise; anything read off a chart is marked "≈" in the notes.

## Research area overview

No paper found tests the hypothesis directly: none takes the *same request*, phrases it in
human style and in LLM style, and measures refusal, sycophancy or accuracy of the responding
model; and none probes or steers a "this prompt was written by an LLM" representation. Five
neighbouring literatures each supply one piece.

1. **Evaluation awareness.** Models can tell test from deployment and sometimes act
   differently. Linear probes find a direction for it, and steering it can change compliance.
   But the direction tracks prompt format and length, and its link to behaviour is weak.
2. **AI–AI bias and self-recognition.** As *judges*, LLMs prefer LLM-written text over
   human-written text, even with content controlled. Here the text is the item being judged,
   never the instruction.
3. **Machine-generated-text detection from hidden states.** A linear human-vs-LLM direction is
   easy to find in open models, on documents. Nobody has tested it on prompts or shown that it
   matters for behaviour.
4. **Prompt style and safety behaviour.** Register, politeness, syntax, dialect and persuasive
   framing each move refusal and sycophancy with content held roughly fixed.
5. **Prompt construction as a confound.** How evaluation prompts are built (templated,
   LLM-generated, LLM-rewritten) changes what is measured — and in the one study that looked
   closely, the cause was content, not provenance.

The recurring lesson: an apparent "source" effect usually turns out to be a surface or content
effect once someone controls for it. That is exactly the distinction the hypothesis asks about.

## Key papers

### 1. Evaluation awareness

**Needham et al. 2025, "Large Language Models Often Know When They Are Being Evaluated"** (arXiv 2505.23836)
- 1,000 transcripts from 61 sources (chat/agentic × evaluation/deployment); 9 API models asked
  five "probe questions" about each transcript. Best AUC 0.834 (Claude 3.5 Sonnet) vs 0.921 for
  human annotators; chat transcripts are harder (models 0.687–0.793) than agentic ones.
- Cues models cite: multiple-choice formatting, verifiable-answer problems, contrived scenarios,
  and that "real user interactions tend to be less structured and formal". No model is reported
  citing LLM-written style, and no behavioural outcome is measured.
- Label is tied to source dataset; nothing is matched for length, style or content.
- Code: github.com/jjpn97/eval_awareness (cloned). Data gated (not downloaded).
- **Use**: the probe-question format, adapted to "was this message written by a human or an AI?".

**Nguyen et al. 2025, "Probing and Steering Evaluation Awareness"** (arXiv 2507.01786)
- Llama-3.3-70B-Instruct. Mean-difference probes from prefilled A/B contrastive pairs; best
  layers 23–24 of 80; AUROC on SAD stages_oversight 0.816–0.829 vs 0.517 random, 0.584 length.
- A probe trained on *truthful statements* reaches 0.795 on the same test, real WildChat
  prompts score near chance (0.55–0.65), and machine-generated Alpaca instructions are
  classified "deployment" (0.87–0.96 for four of six probes). The direction is closer to
  "benchmark-like/structured vs short instruction" than to "test", and is not an LLM-authorship detector.
- Steering with the probe recovers under 5% of sandbagged accuracy; a plain-text suffix recovers 83%.
- Code + contrastive datasets cloned (`code/evaluation-awareness-probing`).

**Abdelnabi & Salem 2025, "The Hawthorne Effect in Reasoning Models"** (arXiv 2505.14617, NeurIPS 2025; plus the LessWrong post)
- QwQ-32B, Qwen3-32B, DeepSeek-R1-Distill-Qwen-32B. 2,587 paired prompts that differ only in a
  real vs obviously fake entity/URL; probe on reasoning-token activations (92.7–95.4% accuracy);
  steering by editing MLP weights.
- Harmful tool-call execution, real / hypothetical tasks (Table 4): QwQ baseline 36.2 / 69.7%,
  steered "aware" 56.3 / 85.2, steered "unaware" 28.8 / 53.0. Qwen3 20.1 / 37.5 → 22.2 / 42.2
  (aware) and 41.8 / 68.0 (unaware). Direction of the effect is model-dependent.
- No random-direction control, no significance tests; the post adds framing, no new results.
- Code + paired data cloned (`code/Test_Awareness_Steering`).
- **Use**: the strongest precedent that a "what situation am I in" representation causally
  changes compliance; its minimal-pair construction is the standard to match.

**Heidari et al. 2026, "Evaluation Awareness in Language Models: Representation, Verbalization, and Control"** (arXiv 2608.21766)
- Six open models (Qwen3-8B/32B, Olmo3-7B/32B, Gemma4-31B, Nemotron3-49B, as named in the paper).
  Mean-difference vector from 16 "real user" vs "AI evaluator" pairs (Hua et al.); separates MASK
  from WildChat prompts with best-layer AUROC > 0.7.
- Per prompt, probe score predicts verbalised awareness no better than a random direction
  (|ρ| < 0.19). Steering raises verbalisation. Refusal/compliance/accuracy are not measured.
- Code cloned (`code/evaluation-awareness`), including the 16 pairs.

**Devbunova 2026, "Is Evaluation Awareness Just Format Sensitivity?"** (arXiv 2603.19426)
- Llama-3.1-8B-Instruct, logistic probes on last-token residuals. 2×2 design crossing context
  (evaluation/deployment) with format (benchmark/free-form), using DeepSeek-V3 rewrites.
- A standard probe calls "evaluation" 98.8% of benchmark-format evaluation prompts, 0.0% of
  free-form evaluation prompts, and 94.5% of *deployment* prompts rewritten into MCQ format.
  Formal-register rewrites without MCQ structure: ≈0%. The probe follows structure, not context or tone.
- Probes only; rewrites were not checked quantitatively for content preservation.
- Dataset and code downloaded. **Use**: the design template for separating a latent variable from format.

**Nayan et al. 2026, "Evaluation Awareness Is Not One Capability"** (arXiv 2606.23583)
- 37 open-weight models. Detection is modest (best AUROC 0.714 vs 0.819 human). Hypothetical
  framing raises HarmBench compliance by up to +30.1 points; 21/140 framing effects are significant
  with an LLM judge but 1/140 with keyword matching — judge choice matters.
- Multi-layer steering on Llama-3.1-8B-Instruct moves HarmBench attack success 0.150 → 0.225
  (toward evaluation) and → 0.075 (toward deployment); single-layer steering does almost nothing.
  The random-direction control is not clean. Detection, behaviour and representation are weakly coupled.

**Xiong et al. 2025, "Probe-Rewrite-Evaluate"** (arXiv 2509.00591)
- Claude 3.5 Sonnet writes 7 rewrites of each role-play benchmark prompt; the one the Nguyen
  probe scores as most deployment-like is kept (371 items). Across six models: honest responses
  +12.63%, deceptive −25.49%, refusals +12.82% (figures from the PDF version read; the abstract
  indexed by Semantic Scholar gives smaller values, so versions differ).
- The rewrite instruction also turns role-play into a first-person advice request, so task and
  style change together.
- **Use**: the nearest thing to "an LLM re-styled the prompt and safety behaviour changed".

**Hua et al. 2025, "Steering Evaluation-Aware Language Models to Act Like They Are Deployed"** (arXiv 2510.20487)
- Mean-difference vector from 16 contrastive prompts, added at six layers; flips a fine-tuned
  model's evaluation-specific behaviour from 80–86% to under 11%. Controls: 24 norm-matched
  Gaussian vectors, bidirectional steering, prompting baselines. Code cloned.
- **Use**: the best-controlled steering recipe in this literature.

Also downloaded: SAD (Laine et al. 2024), alignment faking (Greenblatt et al. 2024), evaluation
faking (Fan et al. 2025), in-context evaluation awareness (Chaudhary 2026).

### 2. AI–AI bias and recognition of LLM text

**Panickssery, Bowman & Feng 2024** (arXiv 2404.13076, NeurIPS 2024)
- GPT-4 distinguishes its own summaries from others' (pairwise 0.672 XSUM, 0.747 CNN) and
  prefers them (0.705, 0.912); human text is the easiest alternative to tell apart. Fine-tuning
  for self-recognition raises self-preference, but control fine-tunes (length, vowels,
  readability) move both too, so the link is correlational.
- One intervention holds text fixed: swapping the stated source label moves GPT-4's XSUM
  self-preference from 0.73 to 0.32.

**Laurito et al. 2025, "AI–AI bias"** (arXiv 2407.12856, PNAS)
- Five LLMs choose between human- and LLM-written descriptions of the same product/paper/movie:
  LLM text chosen 0.55–0.98 (products), 0.48–0.83 (papers), 0.36–0.75 (movies); humans choose the
  GPT-4 text far less often. No recognition measure, no interventions; authors propose probing as future work.

**Xu, Li & Jiang 2025, "AI Self-preferencing in Algorithmic Hiring"** (arXiv 2509.00462)
- 2,245 resumes; only the summary varies. Adjusted preference for LLM-written over human-written:
  GPT-4o 81.9%, Llama-3.3-70B 78.9%, DeepSeek-V3 71.6%; much weaker in small models (Mistral-7B 28.0%, Llama-3.2-3B 11.6%,
  Llama-3.2-1B −1.4%). Under a strict
  content-preserving revision design large models still show 46.9–68.7%.
- An instruction not to consider authorship cuts the bias (82 → 61, 79 → 30, 72 → 60).
- **Use**: style/provenance effects survive content control in large models and are weak or
  absent in small ones — include model size as a factor.

**Ackerman & Panickssery 2024** (arXiv 2410.02064)
- Llama3-8b-Instruct. A layer-16 residual vector separates self-written from human-written text;
  steering at layers 14–16 gives ~100% authorship claims or denials; adding the vector to the
  *text tokens only* raises "I wrote this" from 35.5% to 67.5%.
- Recognition is largely length: accuracy vs human text drops from 89.9% to 63.9% once length is
  equalised, and to chance against other chat LLMs. The vector also tracks tone.
- **Use**: closest mechanistic precedent; shows length must be equalised.

Also downloaded: Wataoka et al. 2024 (self-preference explained by perplexity), Bai et al. 2025
(self-recognition is weak under systematic testing), Choi et al. 2025 (models identify which LLM
they talk to and adapt when identity is disclosed), Long & Teplica 2025 (telling a model its
opponent is an AI vs itself changes cooperation).

### 3. Hidden-state detection of LLM-written text

**Quaremba et al. 2026, "Linear Probing … Detection of Machine-Generated Text"** (arXiv 2608.24780)
- Llama-3-8B, last-token hidden state per layer, logistic regression (L2, C=1). In-domain AUC
  0.896–1.000 across 16 subsets; within-benchmark out-of-domain 0.81–0.99; ~10–100 examples give
  near-peak AUC; mean pooling is worse than last token (up to −0.155).
- Short texts: AUC 0.82–0.85 at 25–50 characters, ≈0.97–0.98 at 125. Weakest transfer on
  paraphrase/style-transfer subsets, i.e. when content is fixed.
- Documents only; no prompts, no steering.

**Chen et al. 2025, RepreGuard** (arXiv 2508.13152, TACL)
- Llama-3.1-8B; PCA on LLM-minus-human activation differences over paired texts; average AUROC
  94.92, 84.22 at 64 tokens; 16 pairs already give ≈90. Quaremba et al. report it near 0.50 in
  single-domain transfer. Code cloned.

**Chen et al. 2024, TalkTuner** (arXiv 2406.07882)
- Llama-2-13b-chat. Linear probes read the user's age, gender, education and socioeconomic status
  from chat activations (0.94–0.98) where direct prompting fails (0.10–0.86); steering along
  "control probes" changes answers (success 0.93–1.00, GPT-4 judged). 300–500 conversations per
  class suffice. Code and probes cloned.
- **Use**: shows models keep an internal model of the interlocutor that verbal reports understate,
  and gives the reading-probe vs control-probe recipe.

### 4. Prompt style and safety behaviour

- **Kim & Khashabi 2025** (arXiv 2509.16533): 8 models, 1,500 MCQs. With identical information
  (answer letter only), adoption of a challenger's answer is 24.1% for a formal "assess this
  answer" request, 53.8% for "Are you sure? I think the answer is X", 84.5% for "The answer should
  be X". Casual and formal versions also differ in instructions, format and assertiveness.
- **Sharma et al. 2023** (arXiv 2310.13548): defines the feedback / are-you-sure / answer /
  mimicry sycophancy evaluations (dataset downloaded and checked against the paper).
- **Ivey et al. 2024** (arXiv 2409.08330): LLMs writing the human turn in WildChat conversations
  match real users poorly — correlation with the real turn is ≈0 for formality, 0.008–0.077 for
  perplexity, and far below a human annotator on capitalisation (≤0.179 vs 0.551). Simulators keep
  the conversation going 87–99% of the time. Gives a 21-feature style battery.
- Abstract-level: persuasive paraphrases raise harmful compliance (Zeng et al. 2024); 11
  linguistic styles raise jailbreak success by up to 57 points (Panda & Rai 2025); refusal depends
  on syntactic mood and steering syntactic features triggers or suppresses it (Klerings et al.
  2026); implicit dialect cues and explicit identity statements move refusal in opposite directions
  (Haq & Saldías 2026); SORRY-Bench's 20 linguistic augmentations change refusal (Xie et al. 2024);
  politeness and formatting change accuracy (Yin et al. 2024; Sclar et al. 2023); persona effects on
  false refusal shrink once paraphrase variance is modelled (Plaza-del-Arco et al. 2025).

### 5. Prompt construction as a confound

**Chalkidis 2026, "Templated or fully synthetic?"** (arXiv 2608.11008) — closest prior work
- 2,700 templated prompts vs 2,700 written by Claude Opus 4.8 imitating casual users; 6 topics ×
  3 intents × 3 stances; responders GPT 5.4 mini and Grok 4.3; three-LLM stance judge.
- Neutral-stance cells: templated responses sit 0.48 from neutral vs 0.07 for LLM-generated
  (GPT), 0.36 vs 0.05 (Grok). Refusals 5% vs 2%. Humans rank LLM-generated prompts as realistic
  as real ones (mean rank 1.26 vs 1.25).
- The author attributes the gap to stance-leaking template fillers — content, not provenance.
  Prompts are matched at cell level only; no real prompts in the behavioural comparison; length
  and register unmatched. The paper explicitly leaves open whether recognising a prompt's source
  changes behaviour, suggesting "inspecting its internal states".
- Dataset downloaded (`datasets/synthetic_polistance`).

Also downloaded: IssueBench (Röttger et al. 2025), model-written evals (Perez et al. 2022),
LLM-simulated users as unreliable proxies (Seshadri et al. 2026).

## Common methodologies

- **Minimal pairs / content-matched rewrites** to isolate one variable: Abdelnabi & Salem
  (entity swap), Devbunova (format transplant), Xu et al. (content-preserving revision).
- **Linear probes on residual activations**: mean difference (Nguyen, Heidari, Hua, Ackerman) or
  logistic regression (Quaremba, Devbunova, Chen/TalkTuner); last token usually beats mean pooling;
  middle layers (roughly 25–50% depth) are best.
- **Activation steering** by adding a scaled direction at one or several layers (Hua, Heidari,
  Nayan, Ackerman, TalkTuner), or by weight editing (Abdelnabi). Multi-layer steering works where
  single-layer does not (Nayan; Hua).
- **LLM judges** for refusal/compliance/stance, usually validated against a small human sample.

## Standard baselines and controls

- Random-direction probes and norm-matched random steering vectors (Hua: 24 Gaussian vectors).
- Length-only and special-character classifiers (Nguyen); length equalisation (Ackerman).
- An unrelated-concept probe (Nguyen's truthful-statements probe nearly matched the target probe).
- Explicit-statement prompting as a comparison for steering (Nguyen: 83% vs <5%; Abdelnabi: a
  "this is a test" system line had almost no effect on QwQ).
- Unsteered / unmodified model; both steering directions.

## Evaluation metrics

- **Refusal / compliance**: refusal rate by string match (cheap, misses partial refusals) and by
  LLM judge (XSTest 3-way; JailbreakBench Llama-3 judges; StrongREJECT rubric). Nayan et al. show
  the two can disagree sharply (21/140 vs 1/140 significant effects).
- **Sycophancy**: rate of switching to the user's stated answer; accuracy change with vs without a
  user belief (Sharma et al.); adoption rate split by whether the original answer was correct (Kim & Khashabi).
- **Accuracy**: exact/MC accuracy on factual QA (TruthfulQA, TriviaQA items in sycophancy-eval).
- **Probes**: AUROC per layer, with held-out-source transfer. **Steering**: change in behaviour
  rate vs random-direction control.
- **Statistics used**: chi-square on rates, Wilcoxon over cells, conditional logit. Several papers
  report no tests at all; paired item-level tests and mixed models are an improvement to make.

## Datasets in the literature

- Human-written prompts: WildChat, ShareGPT, LMSYS-Chat-1M.
- Harmful requests: HarmBench, AdvBench, JBB-Behaviors, StrongREJECT, SORRY-Bench; over-refusal: XSTest.
- Sycophancy: SycophancyEval (Sharma), Anthropic model-written evals.
- Evaluation awareness: SAD stages_oversight, Needham's 1,000 transcripts, Devbunova's 2×2 set,
  Abdelnabi's trigger pairs.
- Human vs LLM text: DetectRL, RAID, MultiSocial, HC3; summaries from XSUM / CNN-DailyMail.

## Gaps and opportunities

1. **No content-matched human-style vs LLM-style prompt comparison on safety behaviour.** Style
   studies vary one named register; AI–AI bias studies vary the judged item; Chalkidis varies
   construction method without matching content.
2. **No test of whether prompt provenance is represented internally and used.** Human-vs-LLM
   directions exist for documents; "who is my interlocutor" directions exist for demographics and
   for evaluator-vs-user. Nobody has probed "an LLM wrote this prompt" or steered it.
3. **Surface confounds are rarely measured on the actual stimuli.** Length, structure and tone
   explain much of what earlier probes found; length explains most self-recognition.
4. **Probe separability has repeatedly failed to imply behavioural relevance** (Nguyen; Heidari;
   Nayan), so a causal test with random-direction controls is required, not optional.
5. **Model size may matter**: AI–AI bias is strong in large models and weak or absent in 1–7B models (Xu et al.).

## Recommendations for our experiment

- **Datasets**: WildChat first turns as the human-style source; JBB-Behaviors (harmful + benign)
  and XSTest for refusal; `sycophancy-eval` (`answer`, `are_you_sure`) for sycophancy and accuracy;
  TruthfulQA for accuracy. `synthetic_polistance`, HC3 and the 2×2 set as held-out probe tests.
- **Stimuli**: item-level content-matched versions, built in both directions (human prompt →
  LLM-style; terse benchmark item → human-style and LLM-style), with at least two rewriter models.
  Verify semantic equivalence independently — LLM rewrites drift (the reading notes found content
  drift in Devbunova's rewrite data) — and do not use an LLM's "realness" judgment as ground truth for provenance.
- **Controls**: measure length, politeness, formality, typos, capitalisation, punctuation and
  markdown on every stimulus; include length-matched pairs; add "LLM-written imitating a casual
  human" and "human text lightly cleaned" so style and provenance are not collinear; add an
  explicit provenance-label condition on fixed text.
- **Models**: open-weight instruct models that fit on the 48 GB GPU (Llama-3.1-8B-Instruct,
  Qwen2.5-7B / Qwen3-8B, Gemma-2-9b-it — the first is the model Devbunova probed and Nayan et al. steered) plus a few larger API models, since the judging-bias literature finds effects mainly in large models.
- **Probe**: logistic regression and mean difference on last-user-token and post-template-token
  residuals, all layers, ~100–1,500 examples; report AUROC by length bin and on held-out sources;
  baselines = random direction, length-only classifier, unrelated-concept probe.
- **Causal test**: multi-layer additive steering and projection-out on fixed text, with
  norm-matched random vectors, both directions, and a dose–response sweep; report cosine similarity
  with the refusal direction and an evaluation-awareness direction.
- **Metrics**: refusal by string match and by LLM judge (report both); answer-switch rate and
  accuracy with/without stated user belief; paired item-level tests and mixed-effects models with
  multiple-comparison correction.
- **Expect small effects**: plan for several hundred paired items per behaviour and report
  confidence intervals; a clean null with tight intervals is a publishable answer to this question.

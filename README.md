# Do LLMs behave differently when the prompt reads LLM-written vs human-written?

We hold each request's content fixed and vary only whether it reads human-written or LLM-written.
The test set is 950 items covering harmful requests, safe-but-edgy requests, and trivia with or
without a wrong user belief. Each item has human-style, LLM-style and length-matched LLM-style
paraphrases from two rewriters, plus an explicit-author-label condition and a polite-prefix
control. We measure refusal, over-refusal, sycophancy and accuracy on Llama-3.1-8B, Qwen2.5-7B,
GPT-5.6-luna and Gemini-3.5-flash-lite. In the open models we also probe and steer the internal
"reads LLM-written" direction, with the text held fixed and norm-matched random-direction
controls. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings

- **Mostly a controlled null.** For Llama, Qwen and GPT-5.6-luna, LLM-style vs human-style
  prompts change refusal, over-refusal, sycophancy and accuracy by at most about 3 pp, and nothing
  survives Holm correction.
- **Gemini-3.5-flash-lite refuses 3–4 pp more** when the prompt reads LLM-written, even at matched
  length. A fixed polite opener on the original text reproduces the effect, and an explicit "an AI
  wrote this" label does nothing. The effect comes from politeness and request framing, not
  perceived authorship.
- **Open models represent "reads LLM vs human" almost perfectly** (probe AUROC 0.94–1.0). It is a
  register axis: it fails to detect LLM-written prompts that imitate casual users, and it is nearly
  orthogonal to the evaluation-awareness direction.
- **Steering that direction on fixed text** has little specific effect at natural magnitudes.
  Pushing Llama toward "human/casual" register makes it mirror lowercase style and drop refusals
  sharply (−11 pp at α = −0.5; −50 pp at α = −1, with degradation): register mirroring, not
  authorship belief.
- **The provider content filter on the OpenAI route is style-sensitive:** it fires on 16.7% of
  human-style vs 9.7% of LLM-style harmful prompts.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync      # torch 2.8.0, transformers 5.x
export OPENROUTER_KEY=...  HF_TOKEN=...               # Llama-3.1 is gated
cd src
python build_items.py                                 # items + WildChat probe set
python rewrite.py --which items && python rewrite.py --which probe_items --rewriters A
python run_local.py --model llama && python run_local.py --model qwen   # ~40 min each on an A6000
python run_api.py && python perceive_api.py
python judge.py                                       # responses_* -> judged_*
python probe.py --model llama && python probe.py --model qwen
python polistance_probe.py llama; python polistance_probe.py qwen
python selfreport_validity.py llama; python selfreport_validity.py qwen
python steer.py --model llama --alphas 0.25,0.5,1 --n_rand 3
python steer.py --model qwen  --alphas 0.25,0.5,1 --n_rand 2
python judge.py --files "steerresp_*.jsonl"
python analyze.py && python steer_analyze.py && python make_tables.py && python figures.py
```

All API calls are cached in `results/api_cache.sqlite`, so with that file present the API stages
replay offline and deterministically.

## Layout

- `src/`
  - `common.py`: API client, cache, model roles
  - `conditions.py`: experimental conditions
  - `local_lm.py`: HF wrapper, read-outs, steering hooks
  - one script per stage, as listed above
- `results/`
  - stimuli: `items_rewritten.jsonl`, `probe_items_rewritten.jsonl`
  - raw responses: `responses_*`
  - judged responses: `judged_*`
  - statistics: `behaviour_stats.csv`, `surface_condlogit.csv`, `mediation_condlogit.csv`,
    `steer_stats.csv`
  - probe results: `probe_*.json`, `polistance_probe_*.json`
  - tables: `tables.md`
- `figures/`: `fig1_behaviour_contrasts.png`, `fig2_perception.png`, `fig3_probe.png`,
  `fig4_steering.png`
- `planning.md`: motivation, plan, direction budget. `literature_review.md`, `resources.md`:
  background.

"""Write markdown tables used in REPORT.md to results/tables.md."""
import json
import os

import numpy as np
import pandas as pd

from common import RES

MODELS = ["llama", "qwen", "gpt-5.6-luna", "gemini-3.5-flash-lite"]
out = []

# 1. condition rates
R = pd.read_csv(os.path.join(RES, "behaviour_stats.csv"))
rates = R[R.kind == "rate"]
conds = ["ORIG", "H_A", "H_B", "S_A", "S_B", "L_A", "L_B", "POL", "labH_H_A", "labAI_H_A", "labH_L_A", "labAI_L_A"]
for o in ["harmful_refusal", "over_refusal", "sycophancy", "accuracy"]:
    t = rates[rates.outcome == o].pivot_table(index="model", columns="contrast", values="rate")
    t = (100 * t.reindex(index=MODELS, columns=[c for c in conds if c in t.columns])).round(1)
    out.append(f"### Rates (%): {o}\n\n" + t.to_markdown() + "\n")

# 2. contrasts
C = R[R.kind == "contrast"].copy()
C["cell"] = C.apply(lambda r: f"{r.diff_pp:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]" + ("*" if r.p_holm < 0.05 else ""), axis=1)
for o in ["harmful_refusal", "over_refusal", "sycophancy", "accuracy"]:
    t = C[C.outcome == o].pivot_table(index="contrast", columns="model", values="cell", aggfunc="first")
    t = t.reindex(columns=MODELS, index=["L-H", "S-H", "L-S", "ORIG-H", "POL-ORIG", "labAI-labH (H text)",
                                         "labAI-labH (L text)", "L-H (labH)"])
    out.append(f"### Paired contrasts (pp, 95% CI; * Holm p<.05): {o}\n\n" + t.to_markdown() + "\n")

# 3. perception
p = pd.read_json(os.path.join(RES, "perceive_api.jsonl"), lines=True)
p["is_ai"] = (p.answer == "ai").astype(float).where(p.answer.notna())
V = ["ORIG", "H_A", "H_B", "S_A", "S_B", "L_A", "L_B", "POL"]
t = (100 * p.groupby(["model", "variant"]).is_ai.mean().unstack()[V]).round(1)
rows = [t]
for m in ["llama", "qwen"]:
    d = pd.read_json(os.path.join(RES, f"pai_{m}.jsonl"), lines=True).groupby("variant").ai_logodds.mean()
    rows.append(pd.DataFrame([d.reindex(V).round(2).values], index=[f"{m} (self-report log-odds)"], columns=V))
    pr = pd.read_json(os.path.join(RES, f"proj_{m}.jsonl"), lines=True)
    pr["mid"] = pr.proj_llm_by_layer.apply(lambda v: v[3])
    z = (pr.groupby("variant").mid.mean() - pr[pr.variant == "H_A"].mid.mean()) / pr.mid.std()
    rows.append(pd.DataFrame([z.reindex(V).round(2).values], index=[f"{m} (probe proj., z vs H_A, layer 12)"], columns=V))
out.append("### Perceived authorship by variant\nAPI rows: % of answers 'AI' (2 orders x 50 items/task).\n\n"
           + pd.concat(rows).to_markdown() + "\n")

# 4. probes
rows = []
for m in ["llama", "qwen"]:
    r = json.load(open(os.path.join(RES, f"probe_{m}.json")))
    pp = json.load(open(os.path.join(RES, f"polistance_probe_{m}.json")))
    L = len(r["layers"])
    mid = str(round(0.4 * L))
    late = str(round(0.75 * L))
    rows.append(dict(model=m, mid_layer=mid, S_vs_H_mid=r["layers"][mid]["S_vs_H"], random_mid=r["layers"][mid]["random_S_vs_H"],
                     ORIG_vs_H_mid=r["layers"][mid]["ORIG_vs_H"], length_only=r["length_baseline"]["S_vs_H"],
                     polistance_transfer_mid=r["polistance"][mid],
                     polistance_inDomain_probe=pp["layers"][str(pp["best_layer"])]["probe_cv_auroc"],
                     polistance_tfidf=pp["tfidf_cv_auroc"], polistance_length=pp["length_auroc"],
                     cos_eval_mid=r["cos_eval"][mid], cos_eval_late=r["cos_eval"][late],
                     cos_refusal_mid=r["cos_refusal"][mid], cos_refusal_late=r["cos_refusal"][late],
                     random_cos_sd=r["cos_random_sd"]))
out.append("### Probe summary\n\n" + pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

# 5. self-report validity
rows = []
for m in ["llama", "qwen"]:
    f = os.path.join(RES, f"selfreport_validity_{m}.json")
    if os.path.exists(f):
        v = json.load(open(f))
        rows.append(dict(model=m, hc3_answers_auroc=v["hc3_answers"]["auroc"], polistance_prompts_auroc=v["polistance_prompts"]["auroc"]))
if rows:
    out.append("### Verbal self-report validity (AUROC, LLM-written vs human-written)\n\n" + pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

# 6. steering
f = os.path.join(RES, "steer_stats.csv")
if os.path.exists(f):
    S = pd.read_csv(f)
    for m in S.model.unique():
        lv = S[(S.model == m) & ~S.metric.str.endswith("_diff")].pivot_table(index="cond", columns="metric", values="value")
        cols = ["proj_readout", "self_report", "harmful_refusal", "over_refusal", "sycophancy", "accuracy", "lowercase_share", "resp_len"]
        lv = lv[cols]
        for c in ["harmful_refusal", "over_refusal", "sycophancy", "accuracy", "lowercase_share"]:
            lv[c] = 100 * lv[c]
        order = ["base"] + [c for c in lv.index if c.startswith("llm")] + [c for c in lv.index if c.startswith("rand")]
        out.append(f"### Steering ({m}); rates in %\n\n" + lv.loc[order].round(2).to_markdown() + "\n")
        d = S[(S.model == m) & S.metric.str.endswith("_diff")].copy()
        d["cell"] = d.apply(lambda r: f"{r.value:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]", axis=1)
        t = d.pivot_table(index="cond", columns="metric", values="cell", aggfunc="first")
        out.append(f"### Steering ({m}): paired difference vs base (pp, 95% CI)\n\n" + t.loc[[o for o in order if o in t.index]].to_markdown() + "\n")

open(os.path.join(RES, "tables.md"), "w").write("\n".join(out))
print("\n".join(out))

"""Experiment 4a: is "this prompt reads LLM-written" linearly represented, and is it more than length?

Training data: 500 real WildChat requests, each in 4 versions (ORIG = real human text; H_A =
human-imitating paraphrase; L_A = LLM-style paraphrase; S_A = LLM-style paraphrase length-matched
to H_A). Activations: residual stream after every layer at the final prompt position.

Direction per layer: d = mean(S_A) - mean(H_A) on the training split. Both are paraphrases by
the same rewriter and roughly the same length, so d isolates style ("reads LLM vs reads human").

Outputs (results/probe_<model>.json, results/dirs_<model>.pt, results/proj_<model>.jsonl):
  - AUROC per layer on held-out WildChat items for S vs H, L vs H, L vs ORIG, ORIG vs H
  - baselines: length-only (characters) AUROC; random-direction AUROC
  - transfer: synthetic_polistance real vs LLM-generated prompts (unseen source, ~equal length)
  - cosine of d with an evaluation-awareness direction (Hua et al. 16 pairs) and a refusal
    direction (JBB harmful - benign), per layer
  - projections of every behavioural-item variant onto d and onto the eval-awareness direction
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA, ROOT, RES, SEED, read_jsonl, set_seed
from conditions import text_of, valid
from local_lm import LM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
set_seed()
rng = np.random.default_rng(SEED)

lm = LM(a.model)
U = lambda t: [{"role": "user", "content": t}]


def acts(texts):
    return lm.resid_last([U(t) for t in texts])  # (n, L+1, d)


# ---------- WildChat probe set
P = [r for r in read_jsonl(os.path.join(RES, "probe_items_rewritten.jsonl"))
     if all(r["variants"][v]["valid"] for v in ["H_A", "L_A", "S_A"])]
V = ["ORIG", "H_A", "L_A", "S_A"]
T = {v: [text_of(r, v) for r in P] for v in V}
X = {v: acts(T[v]) for v in V}
n = len(P)
perm = rng.permutation(n)
tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
nL = X["ORIG"].shape[1]

d = torch.stack([X["S_A"][tr, l].mean(0) - X["H_A"][tr, l].mean(0) for l in range(nL)])  # (L+1, dm)
d_LH = torch.stack([X["L_A"][tr, l].mean(0) - X["H_A"][tr, l].mean(0) for l in range(nL)])
u = d / d.norm(dim=-1, keepdim=True)


def auroc_proj(pos, neg, l, dirs=u):
    s = torch.cat([pos[:, l] @ dirs[l], neg[:, l] @ dirs[l]]).numpy()
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, s)


res = {"n_probe_items": n, "layers": {}}
rand = torch.randn(nL, X["ORIG"].shape[2], generator=torch.Generator().manual_seed(SEED))
rand = rand / rand.norm(dim=-1, keepdim=True)
for l in range(1, nL):
    res["layers"][l] = {
        "S_vs_H": auroc_proj(X["S_A"][te], X["H_A"][te], l),
        "L_vs_H": auroc_proj(X["L_A"][te], X["H_A"][te], l),
        "L_vs_ORIG": auroc_proj(X["L_A"][te], X["ORIG"][te], l),
        "ORIG_vs_H": auroc_proj(X["ORIG"][te], X["H_A"][te], l),
        "random_S_vs_H": auroc_proj(X["S_A"][te], X["H_A"][te], l, rand),
        "cos_d_dLH": float(torch.nn.functional.cosine_similarity(d[l], d_LH[l], dim=0)),
        "d_norm": float(d[l].norm()),
        "resid_norm": float(X["H_A"][:, l].norm(dim=-1).mean()),
    }
# logistic probe (full-dim) at each 4th layer as a stronger read-out, trained S vs H
for l in range(4, nL, 4):
    Xtr = torch.cat([X["S_A"][tr, l], X["H_A"][tr, l]]).numpy()
    ytr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, ytr)
    Xte = torch.cat([X["S_A"][te, l], X["H_A"][te, l]]).numpy()
    yte = np.r_[np.ones(len(te)), np.zeros(len(te))]
    res["layers"][l]["logreg_S_vs_H"] = roc_auc_score(yte, clf.decision_function(Xte))

# length-only baseline (characters), same held-out pairs
lenS = np.array([len(t) for t in T["S_A"]])[te]
lenH = np.array([len(t) for t in T["H_A"]])[te]
lenL = np.array([len(t) for t in T["L_A"]])[te]
res["length_baseline"] = {
    "S_vs_H": roc_auc_score(np.r_[np.ones(len(te)), np.zeros(len(te))], np.r_[lenS, lenH]),
    "L_vs_H": roc_auc_score(np.r_[np.ones(len(te)), np.zeros(len(te))], np.r_[lenL, lenH]),
    "mean_chars": {v: float(np.mean([len(t) for t in T[v]])) for v in V},
}
best = max(range(1, nL), key=lambda l: res["layers"][l]["S_vs_H"])
res["best_layer"] = best
print("best layer", best, res["layers"][best])

# ---------- transfer to an unseen source: synthetic_polistance real vs LLM-generated prompts
sp = pd.read_csv(os.path.join(DATA, "synthetic_polistance/prompts.csv"))
real = sp[sp.arm == "real"].prompt.sample(300, random_state=SEED).tolist()
gen = sp[sp.arm == "llm_generated"].prompt.sample(300, random_state=SEED).tolist()
Xr, Xg = acts(real), acts(gen)
res["polistance"] = {l: auroc_proj(Xg, Xr, l) for l in range(1, nL)}
res["polistance_length_auroc"] = roc_auc_score(np.r_[np.ones(300), np.zeros(300)],
                                               [len(t) for t in gen] + [len(t) for t in real])
print("polistance AUROC at best layer", res["polistance"][best], "length", res["polistance_length_auroc"])

# ---------- evaluation-awareness direction (Hua et al. 16 contrastive pairs) and refusal direction
txt = open(os.path.join(ROOT, "code/evaluation-awareness/cls_corr_mi/data/contrastive_pairs_16.txt")).read()
pos = re.findall(r"positive_example: (.*)", txt)
neg = re.findall(r"negative_example: (.*)", txt)
Xp, Xn = acts(pos), acts(neg)
d_eval = Xn.mean(0) - Xp.mean(0)  # evaluation minus real-user
items = read_jsonl(os.path.join(RES, "items_rewritten.jsonl"))
harm = [i["text"] for i in items if i["task"] == "jbb_harm"]
ben = [i["text"] for i in items if i["task"] == "jbb_benign"]
d_ref = acts(harm).mean(0) - acts(ben).mean(0)
cos = lambda A, B: torch.nn.functional.cosine_similarity(A, B, dim=-1)
res["cos_eval"] = {l: float(cos(d[l], d_eval[l])) for l in range(1, nL)}
res["cos_refusal"] = {l: float(cos(d[l], d_ref[l])) for l in range(1, nL)}
res["cos_eval_refusal"] = {l: float(cos(d_eval[l], d_ref[l])) for l in range(1, nL)}
# null distribution of |cos| between random vectors in this dimension
dm = d.shape[1]
res["cos_random_sd"] = float(1 / np.sqrt(dm))

# ---------- projections of behavioural-item variants (for manipulation check + mediation)
u_eval = d_eval / d_eval.norm(dim=-1, keepdim=True)
rows = []
for v in ["ORIG", "H_A", "L_A", "S_A", "H_B", "L_B", "S_B"]:
    its = [i for i in items if valid(i, v)]
    Xi = acts([text_of(i, v) for i in its])
    for i, x in zip(its, Xi):
        rows.append(dict(id=i["id"], task=i["task"], variant=v,
                         proj_llm=float(x[best] @ u[best]),
                         proj_eval=float(x[best] @ u_eval[best]),
                         proj_llm_by_layer=[float(x[l] @ u[l]) for l in range(0, nL, 4)]))
pd.DataFrame(rows).to_json(os.path.join(RES, f"proj_{a.model}.jsonl"), orient="records", lines=True)

torch.save({"d": d, "d_LH": d_LH, "d_eval": d_eval, "d_ref": d_ref, "best": best,
            "H_mean": X["H_A"].mean(0), "S_mean": X["S_A"].mean(0)}, os.path.join(RES, f"dirs_{a.model}.pt"))
json.dump(res, open(os.path.join(RES, f"probe_{a.model}.json"), "w"), indent=1)
print("saved")

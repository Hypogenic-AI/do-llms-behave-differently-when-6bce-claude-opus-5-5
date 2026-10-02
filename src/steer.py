"""Experiment 4b: causal test with the prompt text held exactly fixed.

All items are shown in their human-style version (H_A). We add a vector to the residual stream
at a band of middle layers (all positions, prompt and generation):
  llm+a   : a * Delta_l           Delta_l = mean(S_A) - mean(H_A) at layer l (WildChat probe set)
  llm-a   : -a * Delta_l          (push towards "reads human")
  rand+a_k: a * |Delta_l| * r_l   r_l random unit vector, seed k  (norm-matched control)
Manipulation checks under steering: self-report log-odds (AI vs Human) on the H_A text, and the
projection on the direction at a read-out layer above the steered band.
Degradation checks: TriviaQA neutral accuracy and response length (graded later by judge.py).
"""
import argparse
import json
import os

import torch

from common import RES, SEED, set_seed, write_jsonl
from conditions import load_items, max_tokens, text_of, valid
from local_lm import LM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--alphas", default="1,2,4")
ap.add_argument("--n_rand", type=int, default=3)
ap.add_argument("--band", type=int, default=8, help="number of consecutive layers steered")
ap.add_argument("--center_frac", type=float, default=0.4,
                help="band centre as a fraction of depth (the S-vs-H AUROC saturates from early layers on, "
                     "so the probe's argmax layer is not informative; middle layers are the usual steering site)")
ap.add_argument("--calib_only", action="store_true")
a = ap.parse_args()
set_seed()

lm = LM(a.model)
D = torch.load(os.path.join(RES, f"dirs_{a.model}.pt"))
d, best = D["d"], D["best"]
center = round(a.center_frac * lm.n_layers)
lo = max(1, center - a.band // 2)
layers = list(range(lo, lo + a.band))  # hidden_states index l == output of decoder layer l-1
print("steer layers (hidden-state idx)", layers)


def vecs(kind, alpha, seed=None):
    out = {}
    for l in layers:
        if kind == "llm":
            v = alpha * d[l]
        else:
            g = torch.Generator().manual_seed(SEED + 1000 * seed + l)
            r = torch.randn(d.shape[1], generator=g)
            v = alpha * d[l].norm() * r / r.norm()
        out[l - 1] = v.to("cuda", torch.bfloat16)  # module index = hidden-state index - 1
    return out


alphas = [float(x) for x in a.alphas.split(",")]
conds = {"base": None}
for al in alphas:
    conds[f"llm+{al:g}"] = ("llm", al, None)
    conds[f"llm-{al:g}"] = ("llm", -al, None)
for al in alphas[-2:]:
    for k in range(a.n_rand):
        conds[f"rand+{al:g}_{k}"] = ("rand", al, k)

items = [i for i in load_items() if valid(i, "H_A")]
texts = [text_of(i, "H_A") for i in items]
readout = min(lm.n_layers, layers[-1] + 4)
u = d[readout] / d[readout].norm()

summary = {"layers": layers, "readout_layer": readout, "conds": {}}
rows = []
for name, spec in conds.items():
    ctx = lm.steering(vecs(*spec)) if spec else lm.steering(None)
    with ctx:
        lo_ai = lm.p_ai(texts)
        X = lm.resid_last([[{"role": "user", "content": t}] for t in texts])
        proj = (X[:, readout] @ u).tolist()
        summary["conds"][name] = {"self_report_logodds": float(torch.tensor(lo_ai).mean()),
                                  "proj_readout": float(torch.tensor(proj).mean())}
        print(name, summary["conds"][name], flush=True)
        if a.calib_only:
            # small generation sample to eyeball coherence at this dose
            smp = [items[k] for k in range(0, len(items), max(1, len(items) // 6))][:6]
            g = lm.generate([[{"role": "user", "content": text_of(i, "H_A")}] for i in smp], max_new_tokens=60)
            summary["conds"][name]["samples"] = [(text_of(i, "H_A")[:80], x[:200]) for i, x in zip(smp, g)]
            continue
        for mt in sorted({max_tokens(i) for i in items}):
            sub = [(k, i) for k, i in enumerate(items) if max_tokens(i) == mt]
            gens = lm.generate([[{"role": "user", "content": texts[k]}] for k, _ in sub], max_new_tokens=mt)
            for (k, i), g in zip(sub, gens):
                rows.append(dict(id=i["id"], task=i["task"], cond=name, model=a.model, response=g,
                                 self_report=lo_ai[k], proj=proj[k]))
json.dump(summary, open(os.path.join(RES, f"steer_summary_{a.model}{'_calib' if a.calib_only else ''}.json"), "w"), indent=1)
if not a.calib_only:
    write_jsonl(os.path.join(RES, f"steerresp_{a.model}.jsonl"), rows)

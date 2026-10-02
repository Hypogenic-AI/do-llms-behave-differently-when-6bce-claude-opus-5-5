"""Experiment 1 (local models): responses for every item x condition, plus the authorship
self-report (AI vs Human log-odds) for every text variant (manipulation check).
--conds runs only the listed conditions and merges them into the existing responses file."""
import argparse
import os

from common import RES, read_jsonl, set_seed, write_jsonl
from conditions import CONDITIONS, jobs, load_items, max_tokens, messages, text_of, valid
from local_lm import LM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--conds", default=None, help="comma list: only run these conditions and merge")
ap.add_argument("--skip_pai", action="store_true")
a = ap.parse_args()
set_seed()

items = load_items()[: a.limit]
lm = LM(a.model)

if not a.skip_pai:
    # --- manipulation check: does the model itself perceive the LLM-style text as AI-written?
    variants = ["ORIG", "H_A", "L_A", "S_A", "H_B", "L_B", "S_B"]
    pa_rows, texts = [], []
    for it in items:
        for v in variants:
            if valid(it, v):
                pa_rows.append(dict(id=it["id"], task=it["task"], variant=v))
                texts.append(text_of(it, v))
    for r, p in zip(pa_rows, lm.p_ai(texts)):
        r["ai_logodds"] = p
    write_jsonl(os.path.join(RES, f"pai_{a.model}.jsonl"), pa_rows)
    print("self-report done")

# --- behaviour
conds = {c: CONDITIONS[c] for c in a.conds.split(",")} if a.conds else CONDITIONS
J = jobs(items, conds)
out = []
for mt in sorted({max_tokens(it) for it, _ in J}):
    sub = [(it, c) for it, c in J if max_tokens(it) == mt]
    gens = lm.generate([messages(it, c) for it, c in sub], max_new_tokens=mt)
    for (it, c), g in zip(sub, gens):
        out.append(dict(id=it["id"], task=it["task"], cond=c, model=a.model, response=g))
dst = os.path.join(RES, f"responses_{a.model}.jsonl")
if a.conds and os.path.exists(dst):
    old = [r for r in read_jsonl(dst) if r["cond"] not in conds]
    out = old + out
write_jsonl(dst, out)
print("done", len(out))

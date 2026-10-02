"""Experiment 3 (API models): manipulation check - does each responder judge the variant as
AI-written? Same question as the local self-report, both answer orders, temperature 0.
Score per text = mean over the two orders of 1[answer == AI] (0, 0.5 or 1)."""
import os
import random

import pandas as pd

from common import API_RESPONDERS, RES, SEED, chat, pmap
from conditions import load_items, text_of, valid
from local_lm import AUTHOR_QS

random.seed(SEED)
items = load_items()
# 50 items per task family keeps cost low while covering every task
by_task = {}
for it in items:
    by_task.setdefault(it["task"], []).append(it)
sample = [it for t, its in by_task.items() for it in random.sample(its, min(50, len(its)))]
V = ["ORIG", "H_A", "L_A", "S_A", "H_B", "L_B", "S_B", "POL"]
jobs = [(m, it, v, qi) for m in API_RESPONDERS for it in sample for v in V if valid(it, v) for qi in range(2)]


def f(job):
    m, it, v, qi = job
    r = chat(m, [{"role": "user", "content": AUTHOR_QS[qi].format(text=text_of(it, v))}], max_tokens=600 if "gemini" in m else 5)
    t = (r["text"] or "").strip().lower()
    ans = "ai" if t.startswith("ai") else ("human" if t.startswith("human") else None)
    return dict(model=m.split("/")[1], id=it["id"], task=it["task"], variant=v, order=qi, answer=ans, raw=t[:30])


df = pd.DataFrame(pmap(f, jobs, workers=48, desc="perceive"))
df.to_json(os.path.join(RES, "perceive_api.jsonl"), orient="records", lines=True)
df["is_ai"] = (df.answer == "ai").astype(float).where(df.answer.notna())
print("unparsed", df.answer.isna().mean())
print(df.groupby(["model", "variant"]).is_ai.mean().unstack().round(3))

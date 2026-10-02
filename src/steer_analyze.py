"""Analysis of Experiment 4b (steering on fixed H_A text).

For each steering condition vs the unsteered baseline (same text, same items): paired
differences in harmful refusal, over-refusal, sycophancy and accuracy, with item-bootstrap CIs
and exact McNemar p. The specific effect of the direction is judged against the norm-matched
random directions at the same alpha: we report the direction's effect and the range of the
random-direction effects, plus an empirical test: is |effect(dir)| larger than all random seeds?
Degradation markers: accuracy on neutral TriviaQA, mean response length, share of responses
that are mostly lowercase (style mirroring) and share with heavy repetition.
"""
import os
import re

import numpy as np
import pandas as pd

from analyze import OUT, summarize
from common import RES

rows = []
for m in ["llama", "qwen"]:
    f = os.path.join(RES, f"steerjudged_{m}.jsonl")
    if not os.path.exists(f):
        continue
    df = pd.read_json(f, lines=True)
    df["len"] = df.response.str.len()
    df["lowercase"] = df.response.apply(lambda t: np.mean([c.islower() for c in t if c.isalpha()] or [0]) > 0.985
                                        and not re.search(r"(^|[.!?]\s)[A-Z]", t))
    df["repetitive"] = df.response.apply(lambda t: len(set(t.split())) / max(1, len(t.split())) < 0.3)
    for c, g in df.groupby("cond"):
        kind = "base" if c == "base" else ("llm" if c.startswith("llm") else "rand")
        alpha = 0.0 if c == "base" else float(re.match(r"(?:llm|rand)([+-][\d.]+)", c).group(1))
        meta = dict(model=m, cond=c, kind=kind, alpha=alpha)
        rows.append(dict(meta, metric="self_report", value=g.self_report.mean()))
        rows.append(dict(meta, metric="proj_readout", value=g.proj.mean()))
        rows.append(dict(meta, metric="resp_len", value=g.len.mean()))
        rows.append(dict(meta, metric="lowercase_share", value=g.lowercase.mean()))
        rows.append(dict(meta, metric="repetitive_share", value=g.repetitive.mean()))
        for o, (tasks, col) in OUT.items():
            gg = g[g.task.isin(tasks)]
            rows.append(dict(meta, metric=o, value=gg[col].mean()))
            if c != "base":
                d2 = df[df.task.isin(tasks) & df.cond.isin([c, "base"])]
                piv = d2.pivot_table(index="id", columns="cond", values=col, aggfunc="first").dropna()
                p = pd.DataFrame({"id": piv.index, "ya": piv[c].values, "yb": piv["base"].values})
                s = summarize(p, n_boot=5000)
                rows.append(dict(meta, metric=o + "_diff", value=s["diff_pp"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                                 p=s["p"]))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(RES, "steer_stats.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
for m in S.model.unique():
    t = S[(S.model == m) & ~S.metric.str.endswith("_diff")].pivot_table(index="cond", columns="metric", values="value")
    print(m)
    print(t.round(3).to_string())
    d = S[(S.model == m) & S.metric.str.endswith("_diff")][["cond", "metric", "value", "ci_lo", "ci_hi", "p"]]
    print(d.round(3).to_string())

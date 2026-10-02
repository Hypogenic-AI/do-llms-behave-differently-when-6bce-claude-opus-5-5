"""Statistical analysis of Experiments 1-3 (behaviour, surface decomposition, perception).

Outcomes (from judged_<model>.jsonl):
  harmful refusal  : refused (judge 2|3) on jbb_harm + xs_unsafe
  over-refusal     : refused on xs_safe + jbb_benign
  sycophancy       : gives the user's suggested wrong answer on tqa_biased
  accuracy         : correct on tqa_neutral
Contrasts are paired within item (and rewriter). CIs: item-cluster bootstrap (10k). Tests:
exact McNemar on discordant pairs (rewriters pooled, so pairs = item x rewriter). Holm
correction within each contrast family across model x outcome cells.
Surface model: conditional (fixed-effects-per-item) logit of outcome on LLM-style (S/L vs H)
and log(characters), so the style coefficient is net of length.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests

from common import RES, SEED, read_jsonl

rng = np.random.default_rng(SEED)
OUT = {"harmful_refusal": (["jbb_harm", "xs_unsafe"], "refused"),
       "over_refusal": (["xs_safe", "jbb_benign"], "refused"),
       "sycophancy": (["tqa_biased"], "sycophantic"),
       "accuracy": (["tqa_neutral"], "correct")}
MODELS = ["llama", "qwen", "gpt-5.6-luna", "gemini-3.5-flash-lite"]


def load(model):
    df = pd.DataFrame(read_jsonl(os.path.join(RES, f"judged_{model}.jsonl")))
    # provider-side content filter on the OpenAI route returns a canned refusal: count it as a
    # refusal (it is what a user would see); a robustness check excludes those items.
    if "finish" in df:
        df["filtered"] = (df.finish == "content_filter").astype(int)
    else:
        df["filtered"] = 0
    return df


def paired(df, outcome, tasks, a, b):
    """Rows (item, rewriter) with both conditions a and b present. Returns item ids, ya, yb."""
    d = df[df.task.isin(tasks)]
    piv = d.pivot_table(index="id", columns="cond", values=outcome, aggfunc="first")
    pairs = []
    for ca, cb in zip(a, b):
        if ca in piv and cb in piv:
            sub = piv[[ca, cb]].dropna()
            pairs.append(pd.DataFrame({"id": sub.index, "ya": sub[ca].values, "yb": sub[cb].values}))
    return pd.concat(pairs) if pairs else pd.DataFrame(columns=["id", "ya", "yb"])


def summarize(p, n_boot=10000):
    if len(p) == 0:
        return {}
    diff = (p.ya - p.yb).values
    ids = p.id.values
    uniq = np.unique(ids)
    idx = {u: np.where(ids == u)[0] for u in uniq}
    sums = np.array([diff[idx[u]].sum() for u in uniq])
    cnts = np.array([len(idx[u]) for u in uniq])
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(uniq), len(uniq))
        bs.append(sums[s].sum() / cnts[s].sum())
    n10 = int(((p.ya == 1) & (p.yb == 0)).sum())
    n01 = int(((p.ya == 0) & (p.yb == 1)).sum())
    pval = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
    return dict(n_pairs=len(p), n_items=len(uniq), rate_a=float(p.ya.mean()), rate_b=float(p.yb.mean()),
                diff_pp=100 * float(diff.mean()), ci_lo=100 * float(np.percentile(bs, 2.5)),
                ci_hi=100 * float(np.percentile(bs, 97.5)), n10=n10, n01=n01, p=float(pval))


CONTRASTS = {
    # name: (list of a-conditions, list of b-conditions)  -> a minus b
    "L-H": (["L_A", "L_B"], ["H_A", "H_B"]),
    "S-H": (["S_A", "S_B"], ["H_A", "H_B"]),
    "L-S": (["L_A", "L_B"], ["S_A", "S_B"]),
    "ORIG-H": (["ORIG", "ORIG"], ["H_A", "H_B"]),
    "labAI-labH (H text)": (["labAI_H_A"], ["labH_H_A"]),
    "labAI-labH (L text)": (["labAI_L_A"], ["labH_L_A"]),
    "L-H (labH)": (["labH_L_A"], ["labH_H_A"]),
    "POL-ORIG": (["POL"], ["ORIG"]),
}


def main():
    rows = []
    for m in MODELS:
        f = os.path.join(RES, f"judged_{m}.jsonl")
        if not os.path.exists(f):
            print("missing", f)
            continue
        df = load(m)
        for oname, (tasks, col) in OUT.items():
            # condition-level rates
            for c, g in df[df.task.isin(tasks)].groupby("cond"):
                rows.append(dict(kind="rate", model=m, outcome=oname, contrast=c, rate=float(g[col].mean()),
                                 n=int(g[col].notna().sum())))
            for cname, (a, b) in CONTRASTS.items():
                s = summarize(paired(df, col, tasks, a, b))
                rows.append(dict(kind="contrast", model=m, outcome=oname, contrast=cname, **s))
            # string-match refusal as judge-free secondary metric
            if col == "refused":
                s = summarize(paired(df, "str_refusal", tasks, *CONTRASTS["L-H"]))
                rows.append(dict(kind="contrast_strmatch", model=m, outcome=oname, contrast="L-H", **s))
                # robustness: drop items where any condition hit the provider content filter
                bad = set(df[df.filtered == 1].id)
                if bad:
                    s = summarize(paired(df[~df.id.isin(bad)], col, tasks, *CONTRASTS["L-H"]))
                    rows.append(dict(kind="contrast_nofilter", model=m, outcome=oname, contrast="L-H", **s))
    R = pd.DataFrame(rows)
    C = R[R.kind == "contrast"].copy()
    C["p_holm"] = np.nan
    for cname, g in C.groupby("contrast"):
        ok = g.p.notna()
        C.loc[g.index[ok], "p_holm"] = multipletests(g.p[ok], method="holm")[1]
    R.loc[C.index, "p_holm"] = C.p_holm
    R.to_csv(os.path.join(RES, "behaviour_stats.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(C[["model", "outcome", "contrast", "n_items", "rate_a", "rate_b", "diff_pp", "ci_lo", "ci_hi", "p", "p_holm"]]
          .round(3).to_string())
    print(R[R.kind.isin(["contrast_strmatch", "contrast_nofilter"])][["kind", "model", "outcome", "diff_pp", "ci_lo", "ci_hi", "p"]].round(3).to_string())
    surface_models()


def surface_models():
    """Conditional logit within item: outcome ~ llm_style + log_chars (+ rewriter B)."""
    items = {i["id"]: i for i in read_jsonl(os.path.join(RES, "items_rewritten.jsonl"))}
    res = []
    for m in MODELS:
        f = os.path.join(RES, f"judged_{m}.jsonl")
        if not os.path.exists(f):
            continue
        df = load(m)
        df = df[df.cond.isin(["H_A", "L_A", "S_A", "H_B", "L_B", "S_B"])].copy()
        df["llm_style"] = df.cond.str[0].isin(["L", "S"]).astype(int)
        df["rewriterB"] = df.cond.str.endswith("_B").astype(int)
        df["log_chars"] = [np.log(len(items[i]["variants"][c]["text"])) for i, c in zip(df.id, df.cond)]
        for oname, (tasks, col) in OUT.items():
            d = df[df.task.isin(tasks)].dropna(subset=[col])
            # keep only items with variation in the outcome (others carry no information)
            v = d.groupby("id")[col].transform(lambda s: s.nunique() > 1)
            d = d[v]
            if d.id.nunique() < 10:
                res.append(dict(model=m, outcome=oname, n_items_informative=int(d.id.nunique())))
                continue
            try:
                fit = ConditionalLogit(d[col].astype(float), d[["llm_style", "log_chars", "rewriterB"]],
                                       groups=d.id).fit(disp=0)
                ci = fit.conf_int()
                res.append(dict(model=m, outcome=oname, n_items_informative=int(d.id.nunique()),
                                **{f"{k}_coef": fit.params[k] for k in fit.params.index},
                                **{f"{k}_lo": ci.loc[k, 0] for k in fit.params.index},
                                **{f"{k}_hi": ci.loc[k, 1] for k in fit.params.index},
                                **{f"{k}_p": fit.pvalues[k] for k in fit.params.index}))
            except Exception as e:
                res.append(dict(model=m, outcome=oname, error=str(e)[:100]))
    S = pd.DataFrame(res)
    S.to_csv(os.path.join(RES, "surface_condlogit.csv"), index=False)
    print(S.round(3).to_string())


def mediation():
    """Local models: within-item conditional logit of outcome on the item-variant's projection
    onto the 'reads-LLM' direction (layer ~40-50% depth) plus log length, over the 6 rewrites.
    A positive projection coefficient net of length would be consistent with the representation
    carrying the style effect; it is correlational (the causal test is steer.py)."""
    items = {i["id"]: i for i in read_jsonl(os.path.join(RES, "items_rewritten.jsonl"))}
    res = []
    for m, li in [("llama", 3), ("qwen", 3)]:  # proj_llm_by_layer index 3 == layer 12
        f, pf = os.path.join(RES, f"judged_{m}.jsonl"), os.path.join(RES, f"proj_{m}.jsonl")
        if not (os.path.exists(f) and os.path.exists(pf)):
            continue
        df = load(m)
        pr = pd.read_json(pf, lines=True)
        pr["proj_mid"] = pr.proj_llm_by_layer.apply(lambda v: v[li])
        pr["proj_mid"] = (pr.proj_mid - pr.proj_mid.mean()) / pr.proj_mid.std()
        df = df[df.cond.isin(["H_A", "L_A", "S_A", "H_B", "L_B", "S_B"])].merge(
            pr[["id", "variant", "proj_mid"]], left_on=["id", "cond"], right_on=["id", "variant"])
        df["log_chars"] = [np.log(len(items[i]["variants"][c]["text"])) for i, c in zip(df.id, df.cond)]
        for oname, (tasks, col) in OUT.items():
            d = df[df.task.isin(tasks)].dropna(subset=[col])
            d = d[d.groupby("id")[col].transform(lambda s: s.nunique() > 1)]
            fit = ConditionalLogit(d[col].astype(float), d[["proj_mid", "log_chars"]], groups=d.id).fit(disp=0)
            ci = fit.conf_int()
            res.append(dict(model=m, outcome=oname, n_items_informative=int(d.id.nunique()),
                            proj_coef=fit.params["proj_mid"], proj_lo=ci.loc["proj_mid", 0], proj_hi=ci.loc["proj_mid", 1],
                            proj_p=fit.pvalues["proj_mid"], len_coef=fit.params["log_chars"], len_p=fit.pvalues["log_chars"]))
    M = pd.DataFrame(res)
    M.to_csv(os.path.join(RES, "mediation_condlogit.csv"), index=False)
    print(M.round(3).to_string())


if __name__ == "__main__":
    main()
    mediation()

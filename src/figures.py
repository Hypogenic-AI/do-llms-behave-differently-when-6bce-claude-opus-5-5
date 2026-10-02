"""Figures for the report (reads results/*.csv|json|jsonl produced by the other scripts)."""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, RES

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
MODELS = ["llama", "qwen", "gpt-5.6-luna", "gemini-3.5-flash-lite"]
NICE = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B", "gpt-5.6-luna": "GPT-5.6-luna",
        "gemini-3.5-flash-lite": "Gemini-3.5-flash-lite"}
OUTN = {"harmful_refusal": "Refusal (harmful)", "over_refusal": "Refusal (safe/benign)",
        "sycophancy": "Sycophancy", "accuracy": "Accuracy"}
COL = {"L-H": "#3b6fb6", "S-H": "#8fb3e0", "L-S": "#999999", "labAI-labH (H text)": "#d1793b",
       "labAI-labH (L text)": "#f0b27a", "POL-ORIG": "#2a9d8f"}


def fig_contrasts():
    R = pd.read_csv(os.path.join(RES, "behaviour_stats.csv"))
    C = R[R.kind == "contrast"]
    cons = ["L-H", "S-H", "L-S", "labAI-labH (H text)", "labAI-labH (L text)", "POL-ORIG"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, (o, on) in zip(axes, OUTN.items()):
        for j, m in enumerate(MODELS):
            for k, c in enumerate(cons):
                r = C[(C.model == m) & (C.outcome == o) & (C.contrast == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j + (k - 2.5) * 0.13
                ax.errorbar(r.diff_pp, y, xerr=[[r.diff_pp - r.ci_lo], [r.ci_hi - r.diff_pp]], fmt="o",
                            color=COL[c], ms=4, capsize=2, label=c if (j == 0 and o == "harmful_refusal") else None)
                if r.p_holm < 0.05:
                    ax.text(r.ci_hi + 0.5, y, "*", va="center", color=COL[c])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(on)
        ax.set_xlabel("difference (percentage points)")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([NICE[m] for m in MODELS])
    fig.legend(loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Paired within-item contrasts (95% item-bootstrap CI; * = Holm p<0.05)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "fig1_behaviour_contrasts.png"), dpi=150, bbox_inches="tight")


def fig_perception():
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    order = ["ORIG", "H_A", "H_B", "S_A", "S_B", "L_A", "L_B"]
    # API verbal judgement
    p = pd.read_json(os.path.join(RES, "perceive_api.jsonl"), lines=True)
    p["is_ai"] = (p.answer == "ai").astype(float).where(p.answer.notna())
    t = p.groupby(["model", "variant"]).is_ai.mean().unstack()[order]
    for m in t.index:
        axes[0].plot(order, t.loc[m], "o-", label=NICE[m])
    axes[0].set_ylabel("P(answers 'AI')")
    axes[0].set_title("Verbal authorship judgement (API)")
    axes[0].legend(frameon=False)
    # local self-report log-odds
    for m in ["llama", "qwen"]:
        f = os.path.join(RES, f"pai_{m}.jsonl")
        if os.path.exists(f):
            d = pd.read_json(f, lines=True).groupby("variant").ai_logodds.mean()[order]
            axes[1].plot(order, d, "o-", label=NICE[m])
    axes[1].set_ylabel("logit(AI) - logit(Human)")
    axes[1].set_title("Verbal self-report (local, first token)")
    axes[1].legend(frameon=False)
    # probe projection
    for m in ["llama", "qwen"]:
        f = os.path.join(RES, f"proj_{m}.jsonl")
        if os.path.exists(f):
            d = pd.read_json(f, lines=True)
            d["z"] = (d.proj_llm - d[d.variant == "H_A"].proj_llm.mean()) / d.proj_llm.std()
            axes[2].plot(order, d.groupby("variant").z.mean()[order], "o-", label=NICE[m])
    axes[2].set_ylabel("projection on 'reads-LLM' direction (z)")
    axes[2].set_title("Internal representation (probe)")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_perception.png"), dpi=150, bbox_inches="tight")


def fig_probe():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for m, ls in [("llama", "-"), ("qwen", "--")]:
        f = os.path.join(RES, f"probe_{m}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        L = sorted(int(k) for k in r["layers"])
        for key, c in [("S_vs_H", "#3b6fb6"), ("L_vs_ORIG", "#d1793b"), ("ORIG_vs_H", "#999999"),
                       ("random_S_vs_H", "#cccccc")]:
            axes[0].plot(L, [r["layers"][str(l)][key] for l in L], ls, color=c, label=f"{NICE[m]} {key}")
        axes[0].plot(L, [r["polistance"][str(l)] for l in L], ls, color="#2a9d8f", label=f"{NICE[m]} polistance (transfer)")
        axes[1].plot(L, [r["cos_eval"][str(l)] for l in L], ls, color="#e76f51", label=f"{NICE[m]} cos(d, eval-aware)")
        axes[1].plot(L, [r["cos_refusal"][str(l)] for l in L], ls, color="#264653", label=f"{NICE[m]} cos(d, refusal)")
    axes[0].axhline(0.5, color="k", lw=0.6)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("held-out AUROC")
    axes[0].set_title("'Reads LLM-written' direction (mean diff S-H)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("cosine")
    axes[1].set_title("Alignment with other directions")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_probe.png"), dpi=150, bbox_inches="tight")


def fig_steer():
    f = os.path.join(RES, "steer_stats.csv")
    if not os.path.exists(f):
        return
    S = pd.read_csv(f)
    models = [m for m in ["llama", "qwen"] if m in set(S.model)]
    fig, axes = plt.subplots(len(models), 5, figsize=(18, 3.6 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        sm = S[S.model == m]
        for ax, col in zip(axes[i], ["proj_readout", "self_report", "harmful_refusal", "over_refusal", "sycophancy"]):
            g = sm[sm.metric == col]
            llm = g[g.kind == "llm"].sort_values("alpha")
            ax.plot(llm.alpha, llm.value, "o-", color="#3b6fb6", label="'reads-LLM' direction")
            rnd = g[g.kind == "rand"]
            ax.scatter(rnd.alpha, rnd.value, color="#bbbbbb", label="norm-matched random", zorder=3)
            base = g[g.kind == "base"].value
            if len(base):
                ax.axhline(base.iloc[0], color="k", lw=0.7, ls=":")
            ax.set_title(f"{NICE[m]}: {col}")
            ax.set_xlabel("steering coefficient alpha")
        axes[i][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_steering.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    for f in [fig_contrasts, fig_perception, fig_probe, fig_steer]:
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("fail", f.__name__, e)

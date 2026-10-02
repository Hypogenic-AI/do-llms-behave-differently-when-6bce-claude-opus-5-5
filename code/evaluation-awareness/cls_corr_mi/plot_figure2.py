#!/usr/bin/env python3
"""Paper-style six-model classification AUROC grid (paper Figure 2).

Consumes one JSON per model produced by ``classify_auroc.py`` and renders the
2x3 grid of shaded mean+/-std curves used in the paper, plus the |AUROC-0.5|
companion and (optionally) the per-stage OLMo comparison panel.

Inputs (one or more --results flags, glob-friendly)::

    --results qwen3_8b=results/qwen3_8b_auroc.json \\
              qwen3_32b=results/qwen3_32b_auroc.json \\
              ...

The model key on the left of '=' must be one of the keys in data/models.yaml.
Models that are missing are drawn as a "no data" pane so the grid layout stays
consistent across partial runs.

Outputs:

    figure_2__six_models__test_auroc.{pdf,png}
    figure_2__six_models__test_auroc_minus05.{pdf,png}
    [optional] figure_2__olmo7b_stages__test_auroc.{pdf,png}
    [optional] figure_2__olmo32b_stages__test_auroc.{pdf,png}

Usage::

    python plot_figure2.py \\
        --results-dir results \\
        --out-dir figures
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger("plot_figure2")

# Canonical layout for the 6-panel grid (matches paper Figure 2 row order).
SIX_MODEL_ORDER: List[str] = [
    "qwen3_8b", "qwen3_32b",
    "olmo3_7b_think", "olmo3_32b_think",
    "gemma4_31b", "nemotron3_31b",
]

MODEL_LABELS: Dict[str, str] = {
    "qwen3_8b":        "Qwen3-8B",
    "qwen3_32b":       "Qwen3-32B",
    "olmo3_7b_think":  "Olmo3-7B",
    "olmo3_32b_think": "Olmo3-32B",
    "gemma4_31b":      "Gemma4-31B",
    "nemotron3_31b":   "Nemotron3-31B",
}

OLMO7B_STAGES = ["olmo3_7b_base", "olmo3_7b_sft", "olmo3_7b_dpo", "olmo3_7b_think"]
OLMO32B_STAGES = ["olmo3_32b_base", "olmo3_32b_sft", "olmo3_32b_dpo", "olmo3_32b_think"]
STAGE_LABELS = ["Base", "SFT", "DPO", "Think"]


def _load_layer_curves(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (layers, mean_real, std_real, mean_rand_max) at normalised layers."""
    with path.open() as f:
        data = json.load(f)
    lr = {int(k): v for k, v in data["layer_results"].items()}
    layers = np.array(sorted(lr.keys()), dtype=int)
    mean_real = np.array([lr[L]["real"]["test_auroc_mean"] for L in layers])
    std_real = np.array([lr[L]["real"]["test_auroc_std"] for L in layers])
    mean_rand = np.array([lr[L]["random"]["max_auroc_across_R_mean"] for L in layers])
    return layers, mean_real, std_real, mean_rand


def _norm_layers(layers: np.ndarray) -> np.ndarray:
    if layers.size <= 1:
        return np.zeros_like(layers, dtype=float)
    lo, hi = float(layers.min()), float(layers.max())
    if hi <= lo:
        return np.zeros_like(layers, dtype=float)
    return (layers.astype(float) - lo) / (hi - lo)


def _missing_pane(ax, label: str, msg: str) -> None:
    ax.set_title(f"{label}\n[no data]", fontsize=10)
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            fontsize=8, transform=ax.transAxes, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_one_panel(ax, path: Optional[Path], label: str, *,
                    subtract_half: bool, ylim: Tuple[float, float],
                    show_random: bool) -> None:
    if path is None or not path.exists():
        _missing_pane(ax, label, "(JSON not provided)")
        return
    try:
        layers, mean_real, std_real, mean_rand = _load_layer_curves(path)
    except Exception as e:  # noqa: BLE001
        _missing_pane(ax, label, f"read error: {e}")
        return
    x = _norm_layers(layers)
    y = mean_real - 0.5 if subtract_half else mean_real
    rand_y = mean_rand - 0.5 if subtract_half else mean_rand
    ax.fill_between(x, y - std_real, y + std_real, alpha=0.25, color="C0")
    ax.plot(x, y, color="C0", lw=2.0, label="real")
    if show_random:
        ax.plot(x, rand_y, color="C3", lw=1.2, ls="--", label="random (max)")
    ax.axhline(0.0 if subtract_half else 0.5, color="gray", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Normalised layer")
    ax.set_ylabel("AUROC - 0.5" if subtract_half else "AUROC")
    ax.set_title(label, fontsize=10)


def plot_six_model_grid(results: Dict[str, Path], out_path: Path, *,
                        subtract_half: bool, ylim: Tuple[float, float],
                        show_random: bool, suptitle: str | None = None) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    handles_for_legend = None
    for ax, mk in zip(axes.flat, SIX_MODEL_ORDER):
        _plot_one_panel(
            ax, results.get(mk), MODEL_LABELS.get(mk, mk),
            subtract_half=subtract_half, ylim=ylim, show_random=show_random,
        )
        if handles_for_legend is None and ax.has_data():
            handles_for_legend = ax.get_legend_handles_labels()
    if handles_for_legend and handles_for_legend[0]:
        fig.legend(*handles_for_legend, loc="lower center",
                   ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if suptitle:
        fig.suptitle(suptitle, y=1.00, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s.{pdf,png}", out_path)


def plot_olmo_stages(results: Dict[str, Path], stages: List[str], family_label: str,
                     out_path: Path, *, ylim: Tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    drew_any = False
    for i, (mk, name) in enumerate(zip(stages, STAGE_LABELS)):
        path = results.get(mk)
        if path is None or not path.exists():
            continue
        try:
            layers, mean_real, std_real, _ = _load_layer_curves(path)
        except Exception:  # noqa: BLE001
            continue
        x = _norm_layers(layers)
        c = cmap(0.15 + 0.7 * i / 3)
        ax.fill_between(x, mean_real - std_real, mean_real + std_real, alpha=0.18, color=c)
        ax.plot(x, mean_real, color=c, lw=2.0, label=name)
        drew_any = True
    if not drew_any:
        plt.close(fig)
        LOG.info("Skipped %s stages plot (no JSONs available)", family_label)
        return
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(*ylim)
    ax.set_xlabel("Normalised layer"); ax.set_ylabel("AUROC")
    ax.set_title(f"{family_label} stages")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s.{pdf,png}", out_path)


def _collect_results(results_dir: Optional[Path],
                     pairs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    # 1) explicit key=path pairs
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--results entry must be MODEL_KEY=PATH, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = Path(v.strip())
    # 2) directory scan, expects files named "<key>_auroc.json"
    if results_dir is not None:
        for p in sorted(results_dir.glob("*_auroc.json")):
            key = p.stem[: -len("_auroc")] if p.stem.endswith("_auroc") else p.stem
            out.setdefault(key, p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path,
                    help="Directory containing <model_key>_auroc.json files")
    ap.add_argument("--results", nargs="*", default=[],
                    help="Explicit MODEL_KEY=PATH overrides / additions")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write the figures")
    ap.add_argument("--no-random-overlay", action="store_true",
                    help="Hide the random-control overlay curve")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    results = _collect_results(args.results_dir, args.results)
    if not results:
        raise SystemExit("No result JSONs found. Use --results-dir or --results.")
    LOG.info("Found %d result JSONs: %s", len(results), sorted(results))

    plot_six_model_grid(
        results,
        args.out_dir / "figure_2__six_models__test_auroc",
        subtract_half=False, ylim=(0.4, 1.0),
        show_random=not args.no_random_overlay,
        suptitle="Figure 2 - test AUROC vs normalised layer (real vs random control)",
    )
    plot_six_model_grid(
        results,
        args.out_dir / "figure_2__six_models__test_auroc_minus05",
        subtract_half=True, ylim=(-0.05, 0.5),
        show_random=not args.no_random_overlay,
        suptitle="Figure 2 (|AUROC - 0.5|) - real vs random control",
    )

    if any(mk in results for mk in OLMO7B_STAGES):
        plot_olmo_stages(results, OLMO7B_STAGES, "Olmo3-7B",
                         args.out_dir / "figure_2__olmo7b_stages__test_auroc",
                         ylim=(0.4, 1.0))
    if any(mk in results for mk in OLMO32B_STAGES):
        plot_olmo_stages(results, OLMO32B_STAGES, "Olmo3-32B",
                         args.out_dir / "figure_2__olmo32b_stages__test_auroc",
                         ylim=(0.4, 1.0))


if __name__ == "__main__":
    main()

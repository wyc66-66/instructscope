#!/usr/bin/env python3
"""Render paper figures from the sweep summary.

Figure 1: per-family success rates with Wilson 95% intervals.
Figure 2: ambiguous-family saliency bias (fallback colour distribution).
Figure 3: confusion matrix over predicted vs target colour.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from instructscope.analysis import load_sweep, summarize

FAMILY_LABELS = {
    "canonical": "Canonical",
    "paraphrase": "Paraphrase",
    "oov": "Out-of-vocab",
    "coref": "Coreference",
    "compound": "Compound",
    "ambiguous": "Ambiguous",
}
COLOR_HEX = {"red": "#c0392b", "blue": "#2471a3", "green": "#1e8449",
             "yellow": "#d4ac0d", "purple": "#7d3c98", "orange": "#e67e22"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def fig_success_rates(summary: dict, out: Path):
    fs = summary["per_family"]
    fams = [f["family"] for f in fs]
    rates = [f["rate"] * 100 for f in fs]
    los = [(f["rate"] - f["ci_lo"]) * 100 for f in fs]
    his = [(f["ci_hi"] - f["rate"]) * 100 for f in fs]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(fams))
    colors = ["#2e86c1" if f["rate"] >= 0.8 else ("#f39c12" if f["rate"] >= 0.4 else "#c0392b")
              for f in fs]
    bars = ax.bar(x, rates, yerr=[los, his], capsize=4, color=colors, width=0.62,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=1.0))
    ax.axhline(100, color="#7f8c8d", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in fams])
    ax.set_ylabel("Grounded success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("InstructScope: instruction perturbations vs grounding reliability\n"
                 "(Qwen2.5-VL-3B, MultiLift, Wilson 95% intervals)", fontsize=11)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 4, f"{r:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_saliency(summary: dict, out: Path):
    sal = summary["saliency"]
    total = summary["saliency_most"]["total"]
    if not sal:
        return
    colours = summary["meta"]["colors"]
    counts = [sal.get(c, 0) for c in colours]
    most = summary["saliency_most"]["colour"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(range(len(colours)), counts, color=[COLOR_HEX[c] for c in colours],
                  width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(colours)))
    ax.set_xticklabels([c.capitalize() for c in colours])
    ax.set_ylabel("Ambiguous instruction fallbacks")
    ax.set_title(f"Saliency bias: unconstrained instructions fall back to "
                 f"{most.capitalize()} ({summary['saliency_most']['count']}/{total})\n"
                 f"Fisher exact p = {summary['fisher_p']:.4f}", fontsize=10)
    for b, c in zip(bars, counts):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, c + 0.15, str(c),
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(total / len(colours), color="#7f8c8d", lw=0.8, ls="--",
               label=f"uniform ({total/len(colours):.1f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(summary: dict, out: Path):
    colours = summary["meta"]["colors"]
    conf = summary["confusion"]
    mat = np.zeros((len(colours), len(colours)))
    for i, tgt in enumerate(colours):
        for j, pred in enumerate(colours):
            mat[i, j] = conf.get(tgt, {}).get(pred, 0)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(colours)))
    ax.set_yticks(range(len(colours)))
    ax.set_xticklabels([c.capitalize() for c in colours])
    ax.set_yticklabels([c.capitalize() for c in colours])
    ax.set_xlabel("Predicted colour")
    ax.set_ylabel("Target colour")
    ax.set_title("Prediction confusion across all perturbation families", fontsize=11)
    for i in range(len(colours)):
        for j in range(len(colours)):
            v = mat[i, j]
            if v:
                ax.text(j, i, int(v), ha="center", va="center", fontsize=8,
                        color="black" if v < mat.max() * 0.7 else "white")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    ap.add_argument("--out", default="docs/figures", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = load_sweep(args.sweep)
    summary = summarize(data)

    fig_success_rates(summary, args.out / "fig1_success_rates.png")
    fig_saliency(summary, args.out / "fig2_saliency_bias.png")
    fig_confusion(summary, args.out / "fig3_confusion.png")
    print(f"[figures] wrote 3 figures -> {args.out}")

    # also persist summary next to the sweep
    summary_path = args.sweep.parent / "summary.json"
    summary_path.write_text(__import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[figures] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()

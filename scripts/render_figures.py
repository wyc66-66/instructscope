#!/usr/bin/env python3
"""Render paper figures from the sweep summary.

Figure 1: per-family success rates with Wilson 95% intervals.
Figure 2: ambiguous-family saliency prior (fallback distribution + the
          phrase-sensitive anchor switch).
Figure 3: confusion matrix over predicted vs target colour.
Figure 4: cross-layout ambiguous fallback across colour->position
          permutations (requires --variants).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
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
    amb = summary["default_analysis"]["ambiguous"]
    colours = summary["meta"]["colors"]
    counts = [amb["counts"].get(c, 0) for c in colours]
    phrases = amb["phrase_anchor"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.7), width_ratios=[1, 1.35])
    fig.subplots_adjust(wspace=0.32)

    # Left: overall fallback distribution vs uniform.
    bars = axL.bar(range(len(colours)), counts, color=[COLOR_HEX[c] for c in colours],
                   width=0.6, edgecolor="black", linewidth=0.6)
    for b, c in zip(bars, counts):
        if c:
            axL.text(b.get_x() + b.get_width() / 2, c + 0.15, str(c),
                     ha="center", va="bottom", fontsize=9)
    axL.axhline(amb["n"] / len(colours), color="#7f8c8d", lw=0.8, ls="--",
                label=f"uniform ({amb['n']/len(colours):.0f})")
    axL.legend(frameon=False)
    axL.set_xticks(range(len(colours)))
    axL.set_xticklabels([c.capitalize() for c in colours])
    axL.set_ylabel("Ambiguous instruction fallbacks")
    axL.set_title(f"Fallback distribution (all {amb['n']} cells)\n"
                  f"χ² vs uniform p = {amb['uniform_p']:.1e}", fontsize=10)

    # Right: per-phrase anchor — same unconstrained intent, different surface
    # form, different deterministic anchor.
    x = np.arange(len(phrases))
    for i, (pp, xi) in enumerate(zip(phrases, x)):
        colour = pp["anchor"]
        axR.bar(xi, pp["count"], width=0.55, color=COLOR_HEX.get(colour, "#999"),
                edgecolor="black", linewidth=0.6)
        axR.text(xi, pp["count"] + 0.1, f"{pp['count']}/{pp['n']}", ha="center",
                 va="bottom", fontsize=9, fontweight="bold")
        axR.text(xi, -0.55, colour.capitalize(), ha="center", va="top", fontsize=8.5,
                 color=COLOR_HEX.get(colour, "#999"))
    axR.set_xticks(x)
    axR.set_xticklabels([p["phrase"] for p in phrases], fontsize=8)
    axR.set_ylabel("Cells falling back to the anchor")
    axR.set_ylim(0, max(p["n"] for p in phrases) + 1.2)
    axR.set_title(f"Phrase-sensitive anchor (Fisher exact p = {amb['phrase_anchor_p']:.1e})",
                  fontsize=10)
    for s in axR.spines.values():
        s.set_visible(False)

    fig.suptitle("InstructScope: under-specification produces a deterministic, phrase-sensitive saliency prior",
                 fontsize=11, y=1.02)
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


def fig_cross_layout(sweep_paths: dict[str, Path], out: Path):
    """Cross-layout ambiguous fallback: does the biased prior survive a colour->position
    permutation? Four deterministic layouts; the *anchor colour* moves with the layout,
    the *biased non-uniform fallback* does not."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(sweep_paths))
    width = 0.2
    colours = ["red", "blue", "green", "yellow"]
    for ci, c in enumerate(colours):
        vals = []
        for name, p in sweep_paths.items():
            data = json.loads(p.read_text(encoding="utf-8"))
            amb = [r for r in data["records"] if r["family"] == "ambiguous"]
            n = sum(1 for r in amb if r["predicted"] == c)
            vals.append(n)
        bars = ax.bar(x + (ci - 1.5) * width, vals, width, color=COLOR_HEX[c],
                      edgecolor="black", linewidth=0.5, label=c.capitalize())
    for i, (name, p) in enumerate(sweep_paths.items()):
        data = json.loads(p.read_text(encoding="utf-8"))
        amb = [r for r in data["records"] if r["family"] == "ambiguous"]
        counts = {c: sum(1 for r in amb if r["predicted"] == c) for c in colours}
        chi2 = sum((counts[c] - len(amb) / 4) ** 2 / (len(amb) / 4) for c in colours)
        up = float(stats.chi2.sf(chi2, df=3))
        ax.text(i, len(amb) + 0.6, f"χ² p = {up:.0e}", ha="center", fontsize=8.5)
    ax.axhline(5, color="#7f8c8d", lw=0.8, ls="--", label="uniform (20/4)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{n}" for n in range(len(sweep_paths))])
    ax.set_ylabel("Ambiguous fallbacks")
    ax.set_ylim(0, 22)
    ax.set_title("Cross-layout: the biased fallback survives a colour–position permutation\n"
                 "the anchor colour moves with the layout; the non-uniform prior does not",
                 fontsize=10.5)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    ap.add_argument("--out", default="docs/figures", type=Path)
    ap.add_argument("--variants", nargs="*", default=[],
                    help="additional sweep.json paths for the cross-layout figure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_sweep(args.sweep)
    summary = summarize(data)

    fig_success_rates(summary, args.out / "fig1_success_rates.png")
    fig_saliency(summary, args.out / "fig2_saliency_bias.png")
    fig_confusion(summary, args.out / "fig3_confusion.png")
    print(f"[figures] wrote 3 figures -> {args.out}")

    if args.variants:
        paths = {"L0": args.sweep}
        for i, v in enumerate(args.variants, start=1):
            paths[f"L{i}"] = Path(v)
        fig_cross_layout(paths, args.out / "fig4_cross_layout.png")
        print(f"[figures] wrote cross-layout figure -> {args.out / 'fig4_cross_layout.png'}")

    # also persist summary next to the sweep
    summary_path = args.sweep.parent / "summary.json"
    summary_path.write_text(__import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[figures] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()

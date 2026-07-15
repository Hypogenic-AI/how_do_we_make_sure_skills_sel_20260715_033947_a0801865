"""Analysis + figures for the Elicitation-Novelty Discriminator study.

Loads results/exp1_results.json (+ exp2, exp3 if present) and produces:
  - results/exp1_summary.csv  : per (model,setting,condition) acc, bootstrap 95% CI
  - results/discriminator.csv : per (model,setting) g_correct, g_selfgen, LCG, SG, stats
  - figures/*.png
Deterministic verifiers => accuracy is a proportion; we use bootstrap CIs and
two-proportion z-tests (with Cohen's h effect size).
"""
import os
import sys
import json
import math
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

SETTING_ORDER = ["arith_b10", "vocab_es", "arith_b9", "arith_b11", "vocab_zeth"]
SETTING_LABEL = {
    "arith_b10": "base-10 mult\n(elicit)", "vocab_es": "Spanish\n(elicit)",
    "arith_b9": "base-9 mult\n(nov-proc)", "arith_b11": "base-11 mult\n(nov-proc)",
    "vocab_zeth": "Zeth vocab\n(nov-info)",
}
COND_ORDER = ["no_skill", "correct_skill", "self_gen_skill", "length_control"]
COND_COLOR = {"no_skill": "#888888", "correct_skill": "#2ca02c",
              "self_gen_skill": "#d62728", "length_control": "#1f77b4"}


def boot_ci(flags, n=5000):
    flags = np.array(flags, dtype=float)
    if len(flags) == 0:
        return (0, 0)
    means = RNG.choice(flags, size=(n, len(flags)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_h(p1, p2):
    def phi(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return 2 * math.asin(math.sqrt(p))
    return phi(p1) - phi(p2)


def two_prop_z(k1, n1, k2, n2):
    """Two-proportion z-test p-value (two-sided)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
    z = (p1 - p2) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def norm_gain(acc, acc0):
    if acc0 >= 1.0:
        return 0.0
    return (acc - acc0) / (1 - acc0)


def analyze_exp1():
    rows = json.loads((RES / "exp1_results.json").read_text())
    idx = {(r["model"], r["setting"], r["condition"]): r for r in rows}

    # summary table
    srows = []
    for r in rows:
        lo, hi = boot_ci(r["correct_flags"])
        srows.append({"model": r["model"], "setting": r["setting"],
                      "regime": r["regime"], "condition": r["condition"],
                      "n": r["n"], "accuracy": round(r["accuracy"], 3),
                      "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    summ = pd.DataFrame(srows)
    summ.to_csv(RES / "exp1_summary.csv", index=False)

    # discriminator metrics
    drows = []
    for (model, setting) in sorted({(r["model"], r["setting"]) for r in rows}):
        base = idx[(model, setting, "no_skill")]
        corr = idx[(model, setting, "correct_skill")]
        selg = idx[(model, setting, "self_gen_skill")]
        leng = idx[(model, setting, "length_control")]
        a0 = base["accuracy"]
        g_c = norm_gain(corr["accuracy"], a0)
        g_s = norm_gain(selg["accuracy"], a0)
        g_l = norm_gain(leng["accuracy"], a0)
        lcg = g_c - g_l                          # length-controlled gain
        sg_util = (g_s / g_c) if abs(g_c) > 1e-6 else float("nan")  # self-gen utility ratio
        drows.append({
            "model": model, "setting": setting, "regime": base["regime"],
            "acc_no_skill": a0, "acc_correct": corr["accuracy"],
            "acc_selfgen": selg["accuracy"], "acc_length": leng["accuracy"],
            "g_correct": round(g_c, 3), "g_selfgen": round(g_s, 3),
            "LCG": round(lcg, 3), "SG_utility": round(sg_util, 3),
            "p_correct_vs_base": round(two_prop_z(corr["n_correct"], corr["n"],
                                                  base["n_correct"], base["n"]), 4),
            "p_selfgen_vs_base": round(two_prop_z(selg["n_correct"], selg["n"],
                                                  base["n_correct"], base["n"]), 4),
            "p_correct_vs_length": round(two_prop_z(corr["n_correct"], corr["n"],
                                                    leng["n_correct"], leng["n"]), 4),
            "h_correct_vs_base": round(cohens_h(corr["accuracy"], a0), 3),
        })
    disc = pd.DataFrame(drows)
    disc.to_csv(RES / "discriminator.csv", index=False)
    return summ, disc, idx


def fig_condition_bars(idx):
    models = sorted({m for (m, s, c) in idx})
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 4.6), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        x = np.arange(len(SETTING_ORDER))
        w = 0.2
        for ci, cond in enumerate(COND_ORDER):
            accs, los, his = [], [], []
            for s in SETTING_ORDER:
                r = idx[(model, s, cond)]
                lo, hi = boot_ci(r["correct_flags"])
                accs.append(r["accuracy"]); los.append(r["accuracy"] - lo); his.append(hi - r["accuracy"])
            ax.bar(x + (ci - 1.5) * w, accs, w, yerr=[los, his], capsize=2,
                   label=cond, color=COND_COLOR[cond])
        ax.set_xticks(x); ax.set_xticklabels([SETTING_LABEL[s] for s in SETTING_ORDER], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy"); ax.set_title(model)
        ax.axhline(0, color="k", lw=0.5)
        if mi == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Exp 1: accuracy by condition across the elicitation→novelty axis", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "exp1_condition_bars.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_discriminator_scatter(disc):
    """Robust discriminator map using absolute accuracy deltas (bounded in [-1,1]),
    which are stable near ceilings unlike the ratio SG_utility.
      x = self-generation delta = acc_selfgen - acc_no_skill  (can the model teach itself? >0 => yes)
      y = content gain          = acc_correct - acc_length     (real content beyond context length)
    Genuine novelty lives at x~0, y>>0 (skill helps a lot, but un-self-generatable)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    regime_marker = {"elicitation": "o", "novelty_procedure": "s", "novelty_information": "^"}
    regime_color = {"elicitation": "#1f77b4", "novelty_procedure": "#ff7f0e",
                    "novelty_information": "#2ca02c"}
    # small label offsets to reduce overlap
    offsets = {}
    for _, r in disc.iterrows():
        x = r["acc_selfgen"] - r["acc_no_skill"]
        y = r["acc_correct"] - r["acc_length"]
        key = (round(x, 2), round(y, 2))
        k = offsets.get(key, 0); offsets[key] = k + 1
        ax.scatter(x, y, s=160, marker=regime_marker.get(r["regime"], "o"),
                   color=regime_color.get(r["regime"], "k"), edgecolor="k", zorder=3, alpha=0.9)
        ax.annotate(f"{r['setting']} ({r['model'][:6]})", (x, y),
                    xytext=(6, 6 + 12 * k), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlim(-0.75, 0.55); ax.set_ylim(-0.2, 1.08)
    ax.set_xlabel("Self-generation delta  acc_selfgen − acc_no_skill   (→ model can teach itself = elicitable)")
    ax.set_ylabel("Content gain  acc_correct − acc_length   (→ real knowledge, not context length)")
    ax.set_title("Discriminator map: genuine novelty = high content gain + zero self-generation")
    ax.annotate("GENUINE NOVELTY\n(skill teaches; model can't self-produce)", (0.02, 1.0),
                fontsize=8, color="#2ca02c", va="top")
    ax.annotate("self-gen HARMS", (-0.72, 0.25), fontsize=8, color="#d62728")
    handles = [plt.Line2D([0], [0], marker=regime_marker[k], color="w",
               markerfacecolor=regime_color[k], markeredgecolor="k", markersize=11, label=k)
               for k in regime_marker]
    ax.legend(handles=handles, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "exp1_discriminator_scatter.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def analyze_exp2():
    p = RES / "exp2_results.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    settings = sorted({d["setting"] for d in data})
    fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4.2), squeeze=False)
    for si, s in enumerate(settings):
        ax = axes[0][si]
        ref = None
        for d in data:
            if d["setting"] != s:
                continue
            iters = [c["iter"] for c in d["curve"]]
            accs = [c["test_acc"] for c in d["curve"]]
            ax.plot(iters, accs, marker="o", label=d["mode"])
            ref = d.get("correct_skill_ref_acc")
        if ref is not None:
            ax.axhline(ref, color="green", ls="--", label="correct-skill ceiling")
        ax.set_title(s); ax.set_xlabel("self-evolution iteration"); ax.set_ylabel("test acc")
        ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=8)
    fig.suptitle("Exp 2: self-evolution learning curves (verifier-only vs with-answer feedback)")
    fig.tight_layout()
    fig.savefig(FIG / "exp2_self_evolution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def analyze_exp3():
    p = RES / "exp3_results.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    conds = ["base_no_skill", "base_in_context", "ft_doc", "ft_examples"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(data)); w = 0.2
    colors = ["#888888", "#2ca02c", "#9467bd", "#8c564b"]
    for ci, c in enumerate(conds):
        vals = [d["results"].get(c, np.nan) for d in data]
        ax.bar(x + (ci - 1.5) * w, vals, w, label=c, color=colors[ci])
    ax.set_xticks(x); ax.set_xticklabels([d["kind"] for d in data])
    ax.set_ylabel("test accuracy (no skill in prompt unless in-context)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp 3: internalizing the skill into weights (Qwen2.5-7B LoRA)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "exp3_internalization.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    summ, disc, idx = analyze_exp1()
    print("=== Discriminator metrics (Exp 1) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(disc.to_string(index=False))
    fig_condition_bars(idx)
    fig_discriminator_scatter(disc)
    analyze_exp2()
    analyze_exp3()
    print("\nFigures written to figures/. CSVs to results/.")


if __name__ == "__main__":
    main()

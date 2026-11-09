#!/usr/bin/env python3
"""
04_inter_llm_agreement.py
=========================
Recomputes every coefficient in the inter-LLM reliability table of the
manuscript (Section 4.9) from the deposited per-review outputs of the two
raters.

    python scripts/04_inter_llm_agreement.py

Input
-----
data/inter_llm_comparison_300.csv -- the stratified sample of 300 reviews
(60 Arabic, 120 French, 120 English), balanced across the four sentiment
classes using the FIRST rater's labels, with the sentiment class, engagement
level and barrier set assigned by each rater. Review text is not
redistributed; reviews are identified by index only.

Barrier sets are stored as semicolon-separated label lists. Labels outside
the nine declared barrier categories are present in the raw output and are
ignored here, exactly as in aqi_core.declared_barriers.

Output
------
A table of Cohen's kappa, raw agreement, Gwet's AC1 and PABAK, each with a
95% percentile bootstrap interval over the 300 sampled reviews, plus the mean
Jaccard similarity of the two raters' barrier sets.

Notes on interpretation, which the manuscript states and which matter when
reading these numbers:

* The sample is balanced on the first rater's sentiment labels, so it is not
  a random draw from the corpus. Kappa computed here characterises agreement
  on a sentiment-balanced sample and does not transfer to the corpus
  distribution.
* Kappa is prevalence-sensitive. For the rare categories (strong engagement,
  high fees, slow loading) it falls towards zero even where raw agreement is
  high; AC1 and PABAK are reported alongside for that reason. For the
  sentiment rows the three coincide, because the balanced design leaves no
  marginal imbalance to correct.
* "high fees" occurs in only two of the 300 reviews, so its kappa is not
  usefully estimable and its interval spans the full range.
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

# A bootstrap resample of a near-constant column can leave the two raters with
# only one label in common, for which kappa is undefined; those resamples are
# dropped below rather than reported.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DECLARED_BARRIERS = [
    "network connectivity", "app crashes", "limited features", "slow loading",
    "security concerns", "high fees", "poor customer service",
    "kyc verification", "failed transactions",
]
N_BOOT = 2000
SEED = 42


def all_labels(cell):
    """Every barrier label in one rater's output for one review."""
    return {part.strip().lower() for part in str(cell or "").split(";") if part.strip()}


def barrier_set(cell):
    """Declared-category barrier labels only (as in aqi_core.declared_barriers)."""
    return {lab for lab in all_labels(cell) if lab in DECLARED_BARRIERS}


def gwet_ac1(x, y):
    """Gwet's AC1 for two raters over a common category set."""
    x, y = np.asarray(x), np.asarray(y)
    cats = sorted(set(x) | set(y))
    q = len(cats)
    if q < 2:
        return float("nan")
    pa = float((x == y).mean())
    pi = {c: ((x == c).mean() + (y == c).mean()) / 2 for c in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (q - 1)
    return (pa - pe) / (1 - pe) if pe != 1 else float("nan")


def pabak(x, y):
    """Prevalence-adjusted bias-adjusted kappa."""
    x, y = np.asarray(x), np.asarray(y)
    q = len(set(x) | set(y))
    if q < 2:
        return float("nan")
    return (q * float((x == y).mean()) - 1) / (q - 1)


def agreement(label, a, b, rng):
    """Point estimates and 95% percentile bootstrap intervals."""
    a, b = list(a), list(b)
    n = len(a)
    idx = np.arange(n)
    boot_k, boot_a, boot_p = [], [], []
    for _ in range(N_BOOT):
        s = rng.choice(idx, n, replace=True)
        aa = [a[i] for i in s]
        bb = [b[i] for i in s]
        try:
            k = cohen_kappa_score(aa, bb)
            if k == k:
                boot_k.append(k)
        except ValueError:
            pass
        for acc, fn in ((boot_a, gwet_ac1), (boot_p, pabak)):
            v = fn(aa, bb)
            if v == v:
                acc.append(v)

    def ci(vals):
        return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (float("nan"),) * 2

    return dict(
        label=label, n=n,
        kappa=cohen_kappa_score(a, b), kappa_ci=ci(boot_k),
        raw=float(np.mean([p == q for p, q in zip(a, b)])),
        ac1=gwet_ac1(a, b), ac1_ci=ci(boot_a),
        pabak=pabak(a, b), pabak_ci=ci(boot_p),
    )


def main(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = np.random.default_rng(SEED)

    g_sent = [r["gemini_sentiment"] for r in rows]
    c_sent = [r["claude_sentiment"] for r in rows]
    g_eng = [r["gemini_inclusion"] for r in rows]
    c_eng = [r["claude_inclusion"] for r in rows]
    lang = [r["lang"] for r in rows]

    results = [agreement("Sentiment, overall", g_sent, c_sent, rng)]
    for code, name in (("ar", "Arabic"), ("fr", "French"), ("en", "English")):
        keep = [i for i, l in enumerate(lang) if l == code]
        results.append(agreement(f"Sentiment, {name}",
                                 [g_sent[i] for i in keep],
                                 [c_sent[i] for i in keep], rng))
    results.append(agreement("Engagement, 4-class", g_eng, c_eng, rng))
    results.append(agreement("Engagement, binary",
                             [0 if v == "none" else 1 for v in g_eng],
                             [0 if v == "none" else 1 for v in c_eng], rng))
    results.append(agreement("Engagement, strong vs rest",
                             [1 if v == "strong" else 0 for v in g_eng],
                             [1 if v == "strong" else 0 for v in c_eng], rng))
    for lab in DECLARED_BARRIERS:
        results.append(agreement(
            f"Barrier: {lab}",
            [1 if lab in barrier_set(r["gemini_barriers"]) else 0 for r in rows],
            [1 if lab in barrier_set(r["claude_barriers"]) else 0 for r in rows], rng))

    hdr = f"{'':30s}{'n':>5s}{'kappa':>8s}{'95% CI':>18s}{'raw':>8s}{'AC1':>8s}{'95% CI':>18s}{'PABAK':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        k_lo, k_hi = r["kappa_ci"]
        a_lo, a_hi = r["ac1_ci"]
        k_ci = "[%.3f, %.3f]" % (k_lo, k_hi)
        a_ci = "[%.3f, %.3f]" % (a_lo, a_hi)
        print(f"{r['label']:30s}{r['n']:5d}{r['kappa']:8.3f}{k_ci:>18s}"
              f"{r['raw']:8.3f}{r['ac1']:8.3f}{a_ci:>18s}{r['pabak']:8.3f}")

    # Mean Jaccard similarity of the two raters' barrier sets, under both label
    # conventions. The manuscript reports the first (complete output); the
    # per-category kappa above use the second (nine declared categories only).
    print()
    for name, fn in (("complete barrier output (reported in the manuscript)", all_labels),
                     ("nine declared categories only", barrier_set)):
        jac = np.array([
            1.0 if not fn(r["gemini_barriers"]) and not fn(r["claude_barriers"])
            else len(fn(r["gemini_barriers"]) & fn(r["claude_barriers"]))
            / len(fn(r["gemini_barriers"]) | fn(r["claude_barriers"]))
            for r in rows])
        boot = [rng.choice(jac, len(jac)).mean() for _ in range(N_BOOT)]
        print(f"Barrier-set Jaccard, {name}: mean {jac.mean():.3f} "
              f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]; "
              f"exact match in {100 * (jac == 1.0).mean():.1f}% of reviews.")

    # Stratification check: the sample is balanced on the FIRST rater's labels.
    from collections import Counter
    print(f"\nFirst rater's sentiment labels in the sample:  {dict(Counter(g_sent))}")
    print(f"Second rater's sentiment labels in the sample: {dict(Counter(c_sent))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inter_llm_comparison_300.csv")
    main(Path(ap.parse_args().data))

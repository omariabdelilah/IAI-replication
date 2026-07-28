#!/usr/bin/env python3
"""
reproduce_manuscript.py
=======================
Regenerates every quantitative result reported in

    Omari, A. & Satauri, I.
    "Beyond Adoption: An Interpretable, Review-Based Index of FinTech Adoption
     Quality - Evidence from Six African Mobile Finance Applications"

from the deposited extracted-signal dataset (data/extracted_signals.csv).

The script recomputes the Inclusive Adoption Index (IAI) from the raw extracted
signals rather than reading the stored iai_score column, so the index formula
itself is verified, not merely echoed.

Note on rounding: the index is rounded to 10 decimal places after aggregation.
The weighted sum 0.50*S + 0.30*I + 0.20*B is subject to binary floating-point
representation error of order 1e-16, which splits mathematically identical
scores into distinct float values. Pearson correlations and group means are
unaffected, but rank-based statistics (Spearman's rho, Kruskal-Wallis H) treat
those spurious differences as tie-breaking information. Rounding restores the
20 distinct index values the scoring scheme actually admits.

Usage
-----
    python scripts/reproduce_manuscript.py
    python scripts/reproduce_manuscript.py --data path/to/extracted_signals.csv

Expected runtime: under one minute.
"""

import argparse
import ast
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

# --------------------------------------------------------------------------
# Scoring scheme (Table 7 of the manuscript)
# --------------------------------------------------------------------------
SENTIMENT_SCORE = {"positive": 1.0, "mixed": 0.5, "neutral": 0.3, "negative": 0.0}
ALPHA, BETA, GAMMA = 0.50, 0.30, 0.20

DECLARED_BARRIERS = {
    "network connectivity", "failed transactions", "poor customer service",
    "kyc verification", "security concerns", "app crashes",
    "slow loading", "limited features", "high fees",
}
RELIABILITY_BARRIERS = {"app crashes", "failed transactions"}

HIGHER = ["mpesa", "wave", "opay"]
LOWER = ["pocket_bank", "cih_pay", "mtn_momo"]

APP_LABEL = {
    "mpesa": "M-Pesa", "wave": "Wave", "opay": "OPay",
    "pocket_bank": "Pocket Bank", "cih_pay": "CIH Pay", "mtn_momo": "MTN MoMo",
}


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check(label, got, expected, tol=5e-4):
    ok = abs(got - expected) <= tol
    flag = "OK " if ok else "!! "
    print(f"  {flag} {label:<52s} {got:>10.4f}   (manuscript {expected})")
    return ok


def parse_list(cell):
    if isinstance(cell, str):
        try:
            value = ast.literal_eval(cell)
            return value if isinstance(value, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


# --------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------
def build_components(df, neutral=0.3, barrier_mode="linear", declared_only=False):
    """Return the three IAI sub-scores S, I, B under a given specification."""
    score_map = dict(SENTIMENT_SCORE, neutral=neutral)
    s = df["sentiment"].map(score_map)
    i = (df["inclusion_signal"] != "none").astype(float)

    if declared_only:
        counts = df["_barriers"].map(lambda lst: sum(1 for x in lst if x in DECLARED_BARRIERS))
    elif barrier_mode == "reliability_x2":
        counts = df["_barriers"].map(
            lambda lst: sum(2 if x in RELIABILITY_BARRIERS else 1 for x in lst)
        )
    else:
        counts = df["_barriers"].map(len)

    if barrier_mode == "binary":
        b = np.where(counts > 0, 0.5, 1.0).astype(float)
    elif barrier_mode == "reciprocal":
        b = 1.0 / (1.0 + counts)
    else:
        b = np.maximum(0.0, 1.0 - 0.25 * counts)

    return s.astype(float), i, pd.Series(np.asarray(b, dtype=float), index=df.index)


def iai(s, i, b, weights=(ALPHA, BETA, GAMMA)):
    a, be, g = weights
    total = a + be + g
    # Rounded to 10 dp: see the note in the module docstring. Without this,
    # floating-point error splits identical scores and biases rank statistics.
    return ((a * s + be * i + g * b) / total).round(10)


def app_means(df, score):
    return score.groupby(df["app"]).mean()


def kendall_vs(baseline_means, other_means):
    apps = sorted(baseline_means.index)
    return stats.kendalltau(baseline_means[apps], other_means[apps]).correlation


def partition_preserved(means):
    top3 = set(means.sort_values(ascending=False).index[:3])
    return top3 == set(HIGHER)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extracted_signals.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    df["_barriers"] = df["adoption_barriers"].map(parse_list)
    df["_cultural"] = df["cultural_factors"].map(parse_list)
    n = len(df)

    header("SAMPLE (Section 3.4, Table 5)")
    print(f"  Analytical sample n = {n:,}   (manuscript 124,953)")
    print(f"  Preprocessed corpus 124,975 - 22 schema failures = {124975 - 22:,}")
    n_mixed = int((df["sentiment"] == "mixed").sum())
    print(f"  Benchmark sample = {n:,} - {n_mixed:,} mixed = {n - n_mixed:,}"
          f"   (manuscript 111,823)")

    s, i, b = build_components(df)
    df["S"], df["I"], df["B"] = s, i, b
    df["IAI"] = iai(s, i, b)
    print(f"  Distinct IAI values = {df['IAI'].nunique()}   (scoring scheme admits 20)")

    header("HEADLINE FIGURES (Sections 4.1, 4.6)")
    check("Mean IAI, full corpus", df["IAI"].mean(), 0.7711)
    check("Review-level Pearson r with star rating",
          df["IAI"].corr(df["rating"]), 0.667)
    check("Review-level Spearman rho",
          df["IAI"].corr(df["rating"], method="spearman"), 0.601)

    # 95% CI via Fisher z
    r = df["IAI"].corr(df["rating"])
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"      95% CI = [{lo:.3f}, {hi:.3f}]   (manuscript [0.664, 0.670])")

    header("TABLE 8 - IAI by application")
    tab8 = pd.DataFrame({
        "Mean": df.groupby("app")["IAI"].mean(),
        "Median": df.groupby("app")["IAI"].median(),
        "Pos%": df.groupby("app")["sentiment"].apply(lambda x: 100 * (x == "positive").mean()),
        "Neg%": df.groupby("app")["sentiment"].apply(lambda x: 100 * (x == "negative").mean()),
        "Engagement%": df.groupby("app")["I"].mean() * 100,
    }).sort_values("Mean", ascending=False)
    tab8.index = [APP_LABEL.get(x, x) for x in tab8.index]
    print(tab8.round(3).to_string())

    header("SECTION 4.2 - Sentiment distribution")
    counts = df["sentiment"].value_counts()
    for label in ["positive", "negative", "mixed", "neutral"]:
        c = int(counts.get(label, 0))
        print(f"  {label:<10s} {c:>7,}  ({100 * c / n:5.1f}%)")

    header("TABLE 11 - Mean IAI by star rating")
    by_star = df.groupby("rating")["IAI"].mean()
    print(by_star.round(3).to_string())
    groups = [g["IAI"].values for _, g in df.groupby("rating")]
    f_stat, p_f = stats.f_oneway(*groups)
    h_stat, p_h = stats.kruskal(*groups)
    grand = df["IAI"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    eta2 = ss_between / ((df["IAI"] - grand) ** 2).sum()
    print(f"  ANOVA F = {f_stat:,.0f} (manuscript 25,270), eta^2 = {eta2:.3f} (0.447)")
    print(f"  Kruskal-Wallis H = {h_stat:,.0f} (manuscript 49,282), "
          f"eps^2 = {(h_stat - 4) / (n - 5):.3f} (0.394)")

    header("TABLE 12 - Inter-component correlations")
    print(df[["S", "I", "B"]].corr().round(3).to_string())

    if sm is not None:
        header("TABLE 13 - Hierarchical regression of star rating on components")
        y = df["rating"].astype(float)
        models = {}
        for name, cols in [("S", ["S"]), ("S+I", ["S", "I"]),
                           ("S+B", ["S", "B"]), ("S+I+B", ["S", "I", "B"])]:
            models[name] = sm.OLS(y, sm.add_constant(df[cols])).fit(cov_type="HC1")
            coefs = {c: round(models[name].params[c], 3) for c in cols}
            print(f"  {name:<7s} R2 = {models[name].rsquared:.3f}   {coefs}")
        d_r2 = models["S+I+B"].rsquared - models["S"].rsquared
        check("Delta R^2 (full vs sentiment-only)", d_r2, 0.019)
        sd = df[["S", "I", "B"]].std()
        betas = {c: round(models['S+I+B'].params[c] * sd[c] / y.std(), 3)
                 for c in ["S", "I", "B"]}
        print(f"      standardised betas {betas}   (manuscript .469 / .086 / .205)")
    else:
        print("\n[statsmodels not installed - Table 13 skipped]")

    header("SECTION 4.7 - Within-stratum (structural) evidence")
    for label in ["mixed", "negative"]:
        sub = df[df["sentiment"] == label].copy()
        sub["bg"] = sub["_barriers"].map(len).clip(upper=3)
        means = sub.groupby("bg")["rating"].mean().round(2).to_dict()
        print(f"  {label:<9s} (n = {len(sub):,})  mean rating by barrier count: {means}")
    print("  engagement, mean star rating with vs. without, by stratum:")
    for label in ["mixed", "negative", "positive", "neutral"]:
        sub = df[df["sentiment"] == label]
        e1, e0 = sub[sub["I"] == 1]["rating"], sub[sub["I"] == 0]["rating"]
        pooled = np.sqrt(((len(e1) - 1) * e1.var() + (len(e0) - 1) * e0.var())
                         / (len(e1) + len(e0) - 2))
        tt = stats.ttest_ind(e1, e0, equal_var=False)
        print(f"    {label:<9s} {e1.mean():.2f} vs {e0.mean():.2f}   "
              f"d = {(e1.mean() - e0.mean()) / pooled:+.2f}   p = {tt.pvalue:.3g}")

    header("SECTION 4.10 - Higher- vs lower-performing groups")
    hi = df[df["app"].isin(HIGHER)]["IAI"]
    lo = df[df["app"].isin(LOWER)]["IAI"]
    pooled = np.sqrt(((len(hi) - 1) * hi.var() + (len(lo) - 1) * lo.var())
                     / (len(hi) + len(lo) - 2))
    check("Higher-performing group mean", hi.mean(), 0.811, tol=1e-3)
    check("Lower-performing group mean", lo.mean(), 0.594, tol=1e-3)
    check("Cohen's d", (hi.mean() - lo.mean()) / pooled, 0.71, tol=5e-3)
    print(f"      Welch t = {stats.ttest_ind(hi, lo, equal_var=False).statistic:.1f} (81.6); "
          f"equal-variance t = {stats.ttest_ind(hi, lo).statistic:.1f} (96.9)")

    header("TABLE 9 - Decomposition of the 2019->2021 change")
    yearly = df.groupby("year")["IAI"].mean()
    print("  Annual mean IAI: " + str(yearly.round(3).to_dict()))
    w19 = df[df["year"] == 2019]["app"].value_counts(normalize=True)
    m21 = df[df["year"] == 2021].groupby("app")["IAI"].mean()
    cf = sum(w19.get(a, 0) * m21.get(a, np.nan) for a in w19.index)
    check("Counterfactual 2021 at fixed 2019 shares", cf, 0.640, tol=1e-3)
    print(f"      raw change {100 * (yearly[2021] / yearly[2019] - 1):+.1f}% "
          f"vs composition-adjusted {100 * (cf / yearly[2019] - 1):+.1f}%")
    print("      within-application 2019->2021:")
    for app in sorted(df["app"].unique()):
        a19 = df[(df["app"] == app) & (df["year"] == 2019)]["IAI"]
        a21 = df[(df["app"] == app) & (df["year"] == 2021)]["IAI"]
        if len(a19) and len(a21):
            print(f"        {APP_LABEL[app]:<12s} {a21.mean() - a19.mean():+.3f}")

    header("FIGURE 6 - Adoption barriers (% of reviews)")
    for label in sorted(DECLARED_BARRIERS,
                        key=lambda k: -sum(1 for lst in df["_barriers"] if k in lst)):
        share = 100 * sum(1 for lst in df["_barriers"] if label in lst) / n
        print(f"  {label:<24s} {share:5.1f}%")
    rel = df["_barriers"].map(lambda l: bool(RELIABILITY_BARRIERS & set(l)))
    both = df["_barriers"].map(lambda l: RELIABILITY_BARRIERS <= set(l))
    print(f"  -> reliability-related: {100 * rel.mean():.1f}% of reviews "
          f"(manuscript 13.1%); both categories: {100 * both.mean():.1f}% (2.1%)")

    header("SECTION 4.5 - Engagement and cultural factors")
    dist = df["inclusion_signal"].value_counts(normalize=True) * 100
    print("  " + str(dist.round(1).to_dict()))
    print(f"  any engagement = {100 * df['I'].mean():.1f}%   (manuscript 78.2%)")
    for k in ["trust issues", "rural usage", "diaspora remittance",
              "cash preference", "language barrier", "family influence"]:
        share = 100 * sum(1 for lst in df["_cultural"] if k in lst) / n
        print(f"  {k:<22s} {share:4.1f}%")

    header("SECTION 3.4 - Non-conforming barrier labels")
    bad = [x for lst in df["_barriers"] for x in lst if x not in DECLARED_BARRIERS]
    affected = df["_barriers"].map(lambda l: any(x not in DECLARED_BARRIERS for x in l))
    print(f"  {len(bad):,} mentions across {len(set(bad))} distinct strings, "
          f"affecting {affected.sum():,} reviews ({100 * affected.mean():.2f}%)")
    print(f"  mean star rating of affected reviews = {df[affected]['rating'].mean():.2f} "
          f"vs {df['rating'].mean():.2f} for the corpus")

    header("SECTION 3.6 - Empirically estimated weights")
    if sm is not None:
        m = sm.OLS(df["rating"].astype(float),
                   sm.add_constant(df[["S", "I", "B"]])).fit()
        w = m.params[["S", "I", "B"]]
        print("  normalised OLS weights: " + str((w / w.sum()).round(3).to_dict())
              + "   (manuscript .488 / .087 / .425)")

    if sm is not None:
        header("SECTION 4.7 - Incremental validity with application fixed effects")
        y = df["rating"].astype(float)
        dummies = pd.get_dummies(df["app"], prefix="app", drop_first=True).astype(float)
        base = sm.OLS(y, sm.add_constant(pd.concat([df[["S"]], dummies], axis=1))).fit(cov_type="HC1")
        full = sm.OLS(y, sm.add_constant(pd.concat([df[["S", "I", "B"]], dummies], axis=1))).fit(cov_type="HC1")
        print(f"  with application fixed effects: R2 {base.rsquared:.3f} -> {full.rsquared:.3f}"
              f"   Delta R^2 = {full.rsquared - base.rsquared:.4f}   (pooled 0.019)")
        print("  Delta R^2 within each application:")
        for app, grp in df.groupby("app"):
            r1 = sm.OLS(grp["rating"].astype(float), sm.add_constant(grp[["S"]])).fit().rsquared
            r2 = sm.OLS(grp["rating"].astype(float), sm.add_constant(grp[["S", "I", "B"]])).fit().rsquared
            print(f"    {APP_LABEL[app]:<12s} n = {len(grp):>6,}   "
                  f"{r1:.3f} -> {r2:.3f}   Delta R^2 = {r2 - r1:.4f}")

    header("TABLE 14 - Component-wise ablation (review level)")
    for name, weights in [
        ("Sentiment only (S)", (1, 0, 0)),
        ("S + I", (0.5, 0.3, 0)),
        ("S + B", (0.5, 0, 0.2)),
        ("Full IAI (S+I+B)", (0.5, 0.3, 0.2)),
    ]:
        score = iai(df["S"], df["I"], df["B"], weights)
        print(f"  {name:<22s} Pearson r = {score.corr(df['rating']):.3f}   "
              f"Spearman rho = {score.corr(df['rating'], method='spearman'):.3f}")

    header("TABLE 6 - Consolidated sensitivity analyses")
    base_means = app_means(df, df["IAI"])
    rows = []

    def add(label, score, subset=None):
        d = df if subset is None else subset
        means = app_means(d, score)
        rows.append((label,
                     kendall_vs(base_means, means),
                     score.corr(d["rating"]),
                     "preserved" if partition_preserved(means) else "CHANGED"))

    s2, i2, b2 = build_components(df, neutral=0.5)
    add("Neutral scored 0.50", iai(s2, i2, b2))
    for mode, name in [("binary", "Barrier penalty: binary"),
                       ("reciprocal", "Barrier penalty: reciprocal"),
                       ("reliability_x2", "Barrier penalty: reliability x2")]:
        s3, i3, b3 = build_components(df, barrier_mode=mode)
        add(name, iai(s3, i3, b3))
    add("Engagement removed (S+B)", iai(df["S"], df["I"], df["B"], (0.5, 0, 0.2)))
    add("Empirical weights", iai(df["S"], df["I"], df["B"], (0.488, 0.087, 0.425)))
    add("Equal weights", iai(df["S"], df["I"], df["B"], (1 / 3, 1 / 3, 1 / 3)))
    add("Barrier-heavy weights", iai(df["S"], df["I"], df["B"], (0.40, 0.20, 0.40)))
    s4, i4, b4 = build_components(df, declared_only=True)
    add("Declared barrier categories only", iai(s4, i4, b4))
    sub = df[df["year"] >= 2021]
    add("Restricted to 2021-2024",
        iai(sub["S"], sub["I"], sub["B"]), subset=sub)

    print(f"  {'Variation':<36s}{'Kendall tau':>12s}{'review r':>10s}   partition")
    print(f"  {'Baseline':<36s}{'---':>12s}{df['IAI'].corr(df['rating']):>10.3f}   ---")
    for label, tau, rr, part in rows:
        print(f"  {label:<36s}{tau:>12.3f}{rr:>10.3f}   {part}")

    header("NOT REPRODUCIBLE FROM THIS FILE")
    print("  The following require the raw review text, which is not redistributed")
    print("  (Google Play terms of service):")
    print("    - language re-derivation (Section 3.2) -> see scripts/01_language_rederivation.py")
    print("    - NLP baseline benchmark, Tables 16-18 -> see scripts/02_nlp_benchmark.py")
    print("    - exact-text duplicate check (Limitation 10)")
    print("  The corrected language labels themselves are deposited in the")
    print("  lang_corrected / lang_method columns of this file.")

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
05_revision3_analyses.py
========================
Every number added in the third revision that is computed from
data/extracted_signals.csv:

    Table 10  decomposition of the change in mean AQI, each year against 2019
    Table 11  within-application change 2019 -> 2021, with n and bootstrap CIs
    Table 12  barrier cluster under six alternative grouping rules
    Table 13  declared adoption-driver frequencies and association with the AQI
    Table 18  star rating regressed on the nine declared barrier indicators
    Section 4.3  application means with 95% CIs; bootstrap CIs for d and delta

    python scripts/05_revision3_analyses.py [--out results_r3.json]

Bootstrap intervals use numpy's default_rng(42) and the same resampling order
as the analyses reported in the paper, so the printed intervals reproduce the
manuscript exactly. Runtime is a few minutes, almost all of it bootstrap.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from aqi_core import (load, build_components, aqi, cliffs_delta, cohens_d,  # noqa: E402
                      parse_list, normalise_label, DECLARED_BARRIERS,
                      HIGHER, LOWER, APP_ORDER, APP_LABEL)

DECLARED_DRIVERS = ["ease of use", "reliability", "transaction speed", "low fees",
                    "access without bank", "merchant payments", "24h availability",
                    "wide agent network", "local language"]
GROUPINGS = [
    ("Narrow (stability only)", ["app crashes"]),
    ("Manuscript", ["app crashes", "failed transactions"]),
    ("Broad", ["app crashes", "failed transactions", "slow loading"]),
    ("Broadest", ["app crashes", "failed transactions", "slow loading", "network connectivity"]),
    ("Transaction failure only", ["failed transactions"]),
    ("Service quality (comparison)", ["poor customer service", "kyc verification", "high fees"]),
    ("Feature gap (comparison)", ["limited features"]),
]


def r(x, d=3):
    return round(float(x), d)


def fixed_composition_mean(frame, weights):
    m = frame.groupby("app")["AQI"].mean()
    w = weights.reindex(m.index).fillna(0.0)
    return float((w / w.sum() * m).sum())


def main(data, out):
    df = load(data)
    s, i, b = build_components(df)
    df["S"], df["I"], df["B"] = s, i, b
    df["AQI"] = aqi(s, i, b)
    y = df["rating"].astype(float)
    R = {}
    rng_a = np.random.default_rng(42)   # Table 10, then d and delta
    rng_b = np.random.default_rng(42)   # Table 11

    # ---------------------------------------------------------------- Table 10
    years = sorted(df["year"].unique())
    w19 = df[df["year"] == years[0]]["app"].value_counts(normalize=True)
    base = float(df[df["year"] == years[0]]["AQI"].mean())
    print("Table 10 - decomposition, each year against the 2019 base")
    print(f"{'year':>5}{'n':>8}  {'actual [95% CI]':24}{'fixed comp. [95% CI]':24}{'difference [95% CI]':24}{'chg actual/fixed'}")
    t10 = []
    for yr in years:
        fr = df[df["year"] == yr]
        raw = float(fr["AQI"].mean()); half = 1.96 * float(fr["AQI"].std()) / np.sqrt(len(fr))
        cf = fixed_composition_mean(fr, w19)
        bc, bg = [], []
        for _ in range(1000):
            sm_ = fr.loc[rng_a.choice(fr.index.values, len(fr), replace=True)]
            c = fixed_composition_mean(sm_, w19); bc.append(c); bg.append(float(sm_["AQI"].mean()) - c)
        row = dict(year=int(yr), n=int(len(fr)), actual=r(raw), actual_ci=[r(raw - half), r(raw + half)],
                   fixed=r(cf), fixed_ci=[r(np.percentile(bc, 2.5)), r(np.percentile(bc, 97.5))],
                   diff=r(raw - cf), diff_ci=[r(np.percentile(bg, 2.5)), r(np.percentile(bg, 97.5))],
                   chg_actual_pct=r(100 * (raw / base - 1), 1), chg_fixed_pct=r(100 * (cf / base - 1), 1))
        t10.append(row)
        print(f"{yr:>5}{len(fr):>8,}  {row['actual']:.3f} {row['actual_ci']!s:17}"
              f"{row['fixed']:.3f} {row['fixed_ci']!s:17}{row['diff']:.3f} {row['diff_ci']!s:17}"
              f"{row['chg_actual_pct']:+.1f}% / {row['chg_fixed_pct']:+.1f}%")
    R["table10"] = t10

    # ----------------------------------------------------- d and delta, Sec 4.3
    hi = df[df["app"].isin(HIGHER)]["AQI"].values
    lo = df[df["app"].isin(LOWER)]["AQI"].values
    bd, bcd = [], []
    for _ in range(2000):
        xa = rng_a.choice(hi, len(hi), replace=True); xb = rng_a.choice(lo, len(lo), replace=True)
        bd.append(cohens_d(xa, xb)); bcd.append(cliffs_delta(xa, xb))
    R["cohens_d"] = [r(cohens_d(hi, lo)), r(np.percentile(bd, 2.5)), r(np.percentile(bd, 97.5))]
    R["cliffs_delta"] = [r(cliffs_delta(hi, lo)), r(np.percentile(bcd, 2.5)), r(np.percentile(bcd, 97.5))]
    print(f"\nCohen's d = {R['cohens_d'][0]:.2f} [{R['cohens_d'][1]:.2f}, {R['cohens_d'][2]:.2f}]; "
          f"Cliff's delta = {R['cliffs_delta'][0]:.2f} [{R['cliffs_delta'][1]:.2f}, {R['cliffs_delta'][2]:.2f}] "
          f"(2,000 bootstrap resamples)")

    # ---------------------------------------------------------------- Table 11
    print("\nTable 11 - within-application change, 2019 -> 2021")
    t11 = []
    for app in APP_ORDER:
        a = df[(df["app"] == app) & (df["year"] == 2019)]["AQI"].values
        c = df[(df["app"] == app) & (df["year"] == 2021)]["AQI"].values
        if not len(a) or not len(c):
            print(f"  {APP_LABEL[app]:12} n2019={len(a):>5}  n2021={len(c):>6}  (no 2019 reviews)")
            t11.append(dict(app=APP_LABEL[app], n2019=int(len(a)), n2021=int(len(c)))); continue
        boots = [rng_b.choice(c, len(c)).mean() - rng_b.choice(a, len(a)).mean() for _ in range(2000)]
        row = dict(app=APP_LABEL[app], n2019=int(len(a)), n2021=int(len(c)), mean2019=r(a.mean()),
                   mean2021=r(c.mean()), change=r(c.mean() - a.mean()),
                   ci=[r(np.percentile(boots, 2.5)), r(np.percentile(boots, 97.5))])
        t11.append(row)
        print(f"  {row['app']:12} n2019={row['n2019']:>5}  n2021={row['n2021']:>6}  "
              f"{row['mean2019']:.3f} -> {row['mean2021']:.3f}  {row['change']:+.3f} {row['ci']}")
    R["table11"] = t11

    # ------------------------------------------------ application means, 4.3
    print("\nApplication means with 95% CIs (Section 4.3)")
    am = []
    for app in APP_ORDER:
        v = df[df["app"] == app]["AQI"].values; h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        am.append(dict(app=APP_LABEL[app], n=int(len(v)), mean=r(v.mean()), ci=[r(v.mean() - h), r(v.mean() + h)]))
        print(f"  {APP_LABEL[app]:12} n={len(v):>6,}  {v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]")
    R["app_means"] = am

    # ---------------------------------------------------------------- Table 18
    BL = sorted(DECLARED_BARRIERS)
    XB = pd.DataFrame({k: df["_barriers"].map(lambda l, k=k: 1.0 if k in l else 0.0) for k in BL}, index=df.index)
    m1 = sm.OLS(y, sm.add_constant(XB)).fit(cov_type="HC1")
    m2 = sm.OLS(y, sm.add_constant(XB.assign(S=df["S"]))).fit(cov_type="HC1")
    m0 = sm.OLS(y, sm.add_constant(df[["S"]])).fit()
    mfull = sm.OLS(y, sm.add_constant(XB.assign(S=df["S"]))).fit()
    print(f"\nTable 18 - star rating on the nine declared barrier indicators (HC1, n = {len(df):,})")
    print(f"  barriers only: R2 = {m1.rsquared:.3f}, intercept {m1.params['const']:.3f}; "
          f"with sentiment: R2 = {m2.rsquared:.3f}, b(S) = {m2.params['S']:.3f}, "
          f"dR2 over S only = {mfull.rsquared - m0.rsquared:.3f}")
    t18 = []
    for k in sorted(BL, key=lambda k: m2.params[k]):
        ci = m1.conf_int().loc[k]
        t18.append(dict(barrier=k, share=r(100 * XB[k].mean(), 2), b=r(m1.params[k]), ci=[r(ci[0]), r(ci[1])],
                        b_adj=r(m2.params[k]), p_adj=float(m2.pvalues[k])))
        print(f"  {k:22} {100 * XB[k].mean():6.2f}%  b={m1.params[k]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
              f"   b|S={m2.params[k]:+.3f}  p={m2.pvalues[k]:.2g}")
    R["table18"] = t18

    # ---------------------------------------------------------------- Table 12
    print("\nTable 12 - barrier cluster under alternative grouping rules (sentiment-adjusted, HC1)")
    t12 = []
    for name, members in GROUPINGS:
        cl = df["_barriers"].map(lambda l, ms=members: any(x in l for x in ms)).astype(float)
        X = pd.DataFrame({"cluster": cl}, index=df.index)
        for k in BL:
            if k not in members:
                X[k] = XB[k]
        X["S"] = df["S"]
        mg = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1"); ci = mg.conf_int().loc["cluster"]
        t12.append(dict(cluster=name, members=members, share=r(100 * cl.mean(), 2),
                        b=r(mg.params["cluster"]), ci=[r(ci[0]), r(ci[1])]))
        print(f"  {name:30} {100 * cl.mean():6.2f}%  b={mg.params['cluster']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    R["table12"] = t12
    R["any_declared_barrier_pct"] = r(100 * df["_barriers"].map(bool).mean(), 1)
    print(f"  at least one declared barrier: {R['any_declared_barrier_pct']}% of reviews")

    # ---------------------------------------------------------------- Table 13
    drv = df["adoption_drivers"].map(parse_list).map(
        lambda l: {normalise_label(x) for x in l if normalise_label(x) in DECLARED_DRIVERS})
    XD = pd.DataFrame({k: drv.map(lambda st, k=k: 1.0 if k in st else 0.0) for k in DECLARED_DRIVERS}, index=df.index)
    md = sm.OLS(df["AQI"], sm.add_constant(XD)).fit(cov_type="HC1")
    hi_m = df["app"].isin(HIGHER); lo_m = df["app"].isin(LOWER)
    print(f"\nTable 13 - declared driver categories (regression of AQI on all nine, HC1, R2 = {md.rsquared:.3f})")
    t13 = []
    for k in DECLARED_DRIVERS:
        ci = md.conf_int().loc[k]
        t13.append(dict(driver=k, n=int(XD[k].sum()), pct=r(100 * XD[k].mean(), 2),
                        higher=r(100 * XD[k][hi_m].mean(), 2), lower=r(100 * XD[k][lo_m].mean(), 2),
                        b=r(md.params[k]), ci=[r(ci[0]), r(ci[1])]))
        print(f"  {k:20} {int(XD[k].sum()):>6,} {100 * XD[k].mean():6.2f}%  "
              f"higher {100 * XD[k][hi_m].mean():5.2f} / lower {100 * XD[k][lo_m].mean():5.2f}  "
              f"b={md.params[k]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    useful = [d for d in DECLARED_DRIVERS if d != "ease of use"]
    R["table13"] = t13
    R["drivers_any_pct"] = r(100 * drv.map(bool).mean(), 1)
    R["drivers_per_review"] = r(drv.map(len).mean(), 2)
    R["tam_ease_of_use_pct"] = r(100 * drv.map(lambda st: "ease of use" in st).mean(), 1)
    R["tam_usefulness_pct"] = r(100 * drv.map(lambda st: any(x in useful for x in st)).mean(), 1)
    print(f"  any declared driver {R['drivers_any_pct']}%, {R['drivers_per_review']} per review; "
          f"ease of use {R['tam_ease_of_use_pct']}%, at least one usefulness driver {R['tam_usefulness_pct']}%")

    if out:
        Path(out).write_text(json.dumps(R, indent=1))
        print(f"\nwritten {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extracted_signals.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.data, a.out)

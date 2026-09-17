#!/usr/bin/env python3
"""
compute_results.py
==================
Computes every index-dependent quantity reported in the manuscript under a
chosen barrier-label specification and writes them to JSON.

    python scripts/compute_results.py                    # baseline (declared)
    python scripts/compute_results.py --labels all       # previous baseline
    python scripts/compute_results.py --raw DIR          # + text-dependent checks

--raw points to the (non-redistributable) review-text files; it is needed only
for the exact-text-duplicate robustness row of Table 7.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).parent))
from aqi_core import *  # noqa: E402,F401,F403


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [r3(np.tanh(z - 1.96 * se)), r3(np.tanh(z + 1.96 * se))]


def exact_perm_p_spearman(x, y):
    import itertools
    x = stats.rankdata(x); y = stats.rankdata(y)
    obs = abs(stats.spearmanr(x, y).correlation)
    cnt = tot = 0
    for p in itertools.permutations(y):
        tot += 1
        if abs(stats.spearmanr(x, p).correlation) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def compute(df, labels="declared", raw_dir=None):
    R = {}
    n = len(df)
    s, i, b = build_components(df, labels=labels)
    df = df.copy()
    df["S"], df["I"], df["B"] = s, i, b
    df["AQI"] = aqi(s, i, b)
    df["SB"] = aqi(s, i, b, (0.5, 0, 0.2))
    lists = df["_barriers"] if labels == "declared" else df["_barriers_raw"]
    df["nb"] = lists.map(len)

    R["n"] = n
    R["distinct_values"] = int(df["AQI"].nunique())
    R["mean_all"] = r3(df["AQI"].mean())
    R["r_review"] = r3(df["AQI"].corr(df["rating"]))
    R["r_review_ci"] = fisher_ci(df["AQI"].corr(df["rating"]), n)
    R["rho_review"] = r3(df["AQI"].corr(df["rating"], method="spearman"))
    R["r_review_SB"] = r3(df["SB"].corr(df["rating"]))
    R["barrier_count_0_4_share"] = r3(100 * (df["nb"] <= 4).mean())
    R["barrier_count_max"] = int(df["nb"].max())

    # ---- Table 8
    g = df.groupby("app")
    t8 = {}
    for a in APP_ORDER:
        d = df[df["app"] == a]
        t8[a] = dict(mean=r3(d["AQI"].mean()), SB=r3(d["SB"].mean()),
                     median=r3(d["AQI"].median()),
                     pos=r3(100 * (d["sentiment"] == "positive").mean()),
                     neg=r3(100 * (d["sentiment"] == "negative").mean()),
                     eng=r3(100 * d["I"].mean()),
                     share_max=r3(100 * (d["AQI"] == 1.0).mean()),
                     share_max_SB=r3(100 * (d["SB"] == 1.0).mean()),
                     mean_rating=r3(d["rating"].mean()),
                     mean_S=r3(d["S"].mean()))
    R["table8"] = t8
    means = g["AQI"].mean()
    R["ranking"] = list(means.sort_values(ascending=False).index)
    R["mean_of_app_means"] = r3(means.mean())
    R["country"] = {c: r3(v) for c, v in df.groupby("country")["AQI"].mean().items()}
    R["tau_SB_vs_base"] = r3(kendall(means, g["SB"].mean()))

    # ---- saturation (R1.2)
    R["share_max_all"] = r3(100 * (df["AQI"] == 1.0).mean())
    R["share_max_SB_all"] = r3(100 * (df["SB"] == 1.0).mean())
    hi = df[df["app"].isin(HIGHER)]; lo = df[df["app"].isin(LOWER)]
    R["share_max_higher"] = r3(100 * (hi["AQI"] == 1.0).mean())
    R["share_max_lower"] = r3(100 * (lo["AQI"] == 1.0).mean())

    # ---- group separation (4.10)
    R["higher_mean"] = r3(hi["AQI"].mean()); R["lower_mean"] = r3(lo["AQI"].mean())
    R["higher_median"] = r3(hi["AQI"].median()); R["lower_median"] = r3(lo["AQI"].median())
    R["cohens_d"] = r3(cohens_d(hi["AQI"], lo["AQI"]))
    R["welch_t"] = r3(stats.ttest_ind(hi["AQI"], lo["AQI"], equal_var=False).statistic)
    cd = cliffs_delta(hi["AQI"].values, lo["AQI"].values)
    R["cliffs_delta"] = r3(cd)
    R["cliffs_delta_ci"] = [r3(v) for v in cliffs_delta_ci(hi["AQI"].values, lo["AQI"].values, 1000)]
    R["prob_superiority"] = r3((cd + 1) / 2)
    R["cohens_d_SB"] = r3(cohens_d(hi["SB"], lo["SB"]))
    R["cliffs_delta_SB"] = r3(cliffs_delta(hi["SB"].values, lo["SB"].values))
    R["higher_mean_SB"] = r3(hi["SB"].mean()); R["lower_mean_SB"] = r3(lo["SB"].mean())
    R["mannwhitney_p"] = float(stats.mannwhitneyu(hi["AQI"], lo["AQI"]).pvalue)

    # ---- Table 11
    R["by_star"] = {int(k): r3(v) for k, v in df.groupby("rating")["AQI"].mean().items()}
    groups = [grp["AQI"].values for _, grp in df.groupby("rating")]
    f_stat, _ = stats.f_oneway(*groups)
    h_stat, _ = stats.kruskal(*groups)
    grand = df["AQI"].mean()
    eta2 = sum(len(x) * (x.mean() - grand) ** 2 for x in groups) / ((df["AQI"] - grand) ** 2).sum()
    R["anova_F"] = round(float(f_stat)); R["anova_eta2"] = r3(eta2)
    R["kw_H"] = round(float(h_stat)); R["kw_eps2"] = r3(h_stat / ((n ** 2 - 1) / (n + 1)))

    # ---- Table 12 + VIF
    c = df[["S", "I", "B"]].corr()
    R["corr_SI"] = r3(c.loc["S", "I"]); R["corr_SB"] = r3(c.loc["S", "B"]); R["corr_IB"] = r3(c.loc["I", "B"])
    X = sm.add_constant(df[["S", "I", "B"]])
    R["vif"] = {k: r3(variance_inflation_factor(X.values, j)) for j, k in enumerate(X.columns) if k != "const"}

    # ---- Table 13
    y = df["rating"].astype(float)
    mods = {}
    for name, cols in [("S", ["S"]), ("S+I", ["S", "I"]), ("S+B", ["S", "B"]), ("S+I+B", ["S", "I", "B"])]:
        m = sm.OLS(y, sm.add_constant(df[cols])).fit(cov_type="HC1")
        mods[name] = m
        R[f"reg_{name}"] = dict(R2=r3(m.rsquared), **{f"b_{k}": r3(m.params[k]) for k in cols})
    R["dR2_SI"] = r3(mods["S+I"].rsquared - mods["S"].rsquared)
    R["dR2_SB"] = r3(mods["S+B"].rsquared - mods["S"].rsquared)
    R["dR2_full"] = r3(mods["S+I+B"].rsquared - mods["S"].rsquared)
    ols_full = sm.OLS(y, sm.add_constant(df[["S", "I", "B"]])).fit()
    ols_s = sm.OLS(y, sm.add_constant(df[["S"]])).fit()
    q, dfr = 2, ols_full.df_resid
    R["nested_F"] = r3(((ols_s.ssr - ols_full.ssr) / q) / (ols_full.ssr / dfr))
    R["nested_df"] = [q, int(dfr)]
    sd = df[["S", "I", "B"]].std()
    R["std_betas"] = {k: r3(mods["S+I+B"].params[k] * sd[k] / y.std()) for k in ["S", "I", "B"]}
    # partial correlations of I and B with rating controlling for S
    for comp in ["I", "B"]:
        ry = sm.OLS(y, sm.add_constant(df[["S"]])).fit().resid
        rx = sm.OLS(df[comp], sm.add_constant(df[["S"]])).fit().resid
        R[f"partial_r_{comp}"] = r3(np.corrcoef(ry, rx)[0, 1])
    w = ols_full.params[["S", "I", "B"]]
    R["empirical_weights"] = {k: r3(v) for k, v in (w / w.sum()).items()}

    # ---- within-stratum (4.7)
    for lab in ["mixed", "negative"]:
        sub = df[df["sentiment"] == lab].copy()
        sub["bg"] = sub["nb"].clip(upper=3)
        R[f"strat_{lab}_means"] = {int(k): r3(v) for k, v in sub.groupby("bg")["rating"].mean().items()}
        gg = [x["rating"].values for _, x in sub.groupby("bg")]
        F, _ = stats.f_oneway(*gg)
        gm = sub["rating"].mean()
        R[f"strat_{lab}_F"] = r3(F)
        R[f"strat_{lab}_eta2"] = r3(sum(len(x) * (x.mean() - gm) ** 2 for x in gg) / ((sub["rating"] - gm) ** 2).sum())
    mixed = df[df["sentiment"] == "mixed"]
    e1, e0 = mixed[mixed["I"] == 1]["rating"], mixed[mixed["I"] == 0]["rating"]
    R["strat_mixed_eng"] = [r3(e1.mean()), r3(e0.mean()), r3(cohens_d(e1, e0))]

    # ---- Table 9 / Figure 2
    yearly = df.groupby("year")["AQI"].agg(["mean", "std", "count"])
    R["yearly_mean"] = {int(k): r3(v) for k, v in yearly["mean"].items()}
    R["yearly_ci_half"] = {int(k): r3(1.96 * r["std"] / np.sqrt(r["count"])) for k, r in yearly.iterrows()}
    w19 = df[df["year"] == 2019]["app"].value_counts(normalize=True)
    m21 = df[df["year"] == 2021].groupby("app")["AQI"].mean()
    cf = sum(w19[a] * m21[a] for a in w19.index)
    R["counterfactual_2021"] = r3(cf)
    R["raw_change_pct"] = r3(100 * (yearly.loc[2021, "mean"] / yearly.loc[2019, "mean"] - 1))
    R["cf_change_pct"] = r3(100 * (cf / yearly.loc[2019, "mean"] - 1))
    R["within_app_19_21"] = {}
    for a in APP_ORDER:
        a19 = df[(df["app"] == a) & (df["year"] == 2019)]["AQI"]
        a21 = df[(df["app"] == a) & (df["year"] == 2021)]["AQI"]
        if len(a19) and len(a21):
            R["within_app_19_21"][a] = r3(a21.mean() - a19.mean())
    d21 = df[df["year"] == 2021]
    R["mpesa_2021_share"] = r3(100 * (d21["app"] == "mpesa").mean())
    R["mpesa_2021_mean"] = r3(d21[d21["app"] == "mpesa"]["AQI"].mean())
    R["wave_share_2019"] = r3(100 * (df[df["year"] == 2019]["app"] == "wave").mean())
    R["wave_share_2021"] = r3(100 * (d21["app"] == "wave").mean())
    R["wave_2021_mean"] = r3(d21[d21["app"] == "wave"]["AQI"].mean())

    # ---- Figure 6 / barriers
    R["barrier_shares"] = {k: r3(100 * lists.map(lambda l: k in l).mean()) for k in sorted(DECLARED_BARRIERS)}
    rel = lists.map(lambda l: bool(RELIABILITY_BARRIERS & set(l)))
    both = lists.map(lambda l: RELIABILITY_BARRIERS <= set(l))
    R["reliability_share"] = r3(100 * rel.mean()); R["reliability_both"] = r3(100 * both.mean())
    for a in ["cih_pay", "mtn_momo"]:
        la = lists[df["app"] == a]
        R[f"{a}_barriers"] = {k: r3(100 * la.map(lambda l: k in l).mean()) for k in sorted(DECLARED_BARRIERS)}

    # ---- Table 10 + Table 14 (application level)
    app = df.groupby("app").agg(AQI=("AQI", "mean"), rating=("rating", "mean"),
                                pos=("sentiment", lambda x: (x == "positive").mean()),
                                neg=("sentiment", lambda x: (x == "negative").mean()))
    for col in ["rating", "pos", "neg"]:
        rr = app["AQI"].corr(app[col])
        R[f"app_r_{col}"] = r3(rr)
        R[f"app_r_{col}_ci"] = fisher_ci(rr, 6)
        R[f"app_r_{col}_p"] = float(stats.pearsonr(app["AQI"], app[col]).pvalue)
        R[f"app_rho_{col}"] = r3(app["AQI"].corr(app[col], method="spearman"))
        R[f"app_rho_{col}_permp"] = r3(exact_perm_p_spearman(app["AQI"].values, app[col].values))
    abl = {}
    for name, wts in [("S", (1, 0, 0)), ("S+I", (0.5, 0.3, 0)), ("S+B", (0.5, 0, 0.2)), ("S+I+B", (0.5, 0.3, 0.2))]:
        sc = aqi(df["S"], df["I"], df["B"], wts)
        am = sc.groupby(df["app"]).mean()
        abl[name] = dict(app_r=r3(am.corr(app["rating"])), app_rho=r3(am.corr(app["rating"], method="spearman")),
                         rev_r=r3(sc.corr(df["rating"])), rev_rho=r3(sc.corr(df["rating"], method="spearman")))
    R["ablation"] = abl

    # ---- Table 6
    rows = {}

    def add(label, score, subset=None):
        d = df if subset is None else subset
        m = score.groupby(d["app"]).mean()
        rows[label] = dict(tau=r3(kendall(means, m)), r=r3(score.corr(d["rating"])),
                           partition=partition_preserved(m),
                           max_mean_diff=r3((m - means.reindex(m.index)).abs().max()))

    s2, i2, b2 = build_components(df, neutral=0.5, labels=labels); add("neutral_050", aqi(s2, i2, b2))
    for mode in ["binary", "reciprocal", "reliability_x2"]:
        s3, i3, b3 = build_components(df, barrier_mode=mode, labels=labels); add(f"pen_{mode}", aqi(s3, i3, b3))
    add("engagement_removed", df["SB"])
    ew = tuple(R["empirical_weights"][k] for k in ["S", "I", "B"])
    add("empirical_weights", aqi(df["S"], df["I"], df["B"], ew))
    add("equal_weights", aqi(df["S"], df["I"], df["B"], (1 / 3, 1 / 3, 1 / 3)))
    add("barrier_heavy", aqi(df["S"], df["I"], df["B"], (0.4, 0.2, 0.4)))
    sub = df[df["year"] >= 2021]; add("years_2021_2024", sub["AQI"], subset=sub)
    other = "all" if labels == "declared" else "declared"
    s5, i5, b5 = build_components(df, labels=other); add(f"labels_{other}", aqi(s5, i5, b5))
    if raw_dir:
        dup = exact_text_duplicate_flags(df, raw_dir)
        R["dup_reviews_sharing_text"] = int(dup["shares"].sum())
        keep = ~dup["drop"]
        R["dup_n_after"] = int(keep.sum()); R["dup_removed"] = int(dup["drop"].sum())
        sub = df[keep.values]
        add("exact_text_dups_removed", sub["AQI"], subset=sub)
        m_sub = sub.groupby("app")["AQI"].mean()
        diff = (m_sub - means)
        R["dup_mean_fall_range"] = [r3(-diff.max()), r3(-diff.min())]
        R["dup_mean_fall_mean"] = r3(-diff.mean())
    R["table6"] = rows
    return R


def exact_text_duplicate_flags(df, raw_dir):
    texts = []
    files = list(Path(raw_dir).glob("*.pkl")) or list(Path(raw_dir).glob("*_FINAL.xlsx"))
    for f in files:
        raw = pd.read_pickle(f) if f.suffix == ".pkl" else pd.read_excel(f)
        t = raw[["review_id", "text"]].drop_duplicates("review_id")
        texts.append(t)
    t = pd.concat(texts).drop_duplicates("review_id").set_index("review_id")["text"]
    txt = df["review_id"].map(t).fillna("").astype(str).map(lambda s: " ".join(s.split()))
    key = df["app"] + "\x00" + txt
    shares = key.duplicated(keep=False)
    drop = key.duplicated(keep="first")
    return pd.DataFrame({"shares": shares.values, "drop": drop.values})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extracted_signals.csv")
    ap.add_argument("--labels", default="declared", choices=["declared", "all"])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = compute(load(a.data), a.labels, a.raw)
    out = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(out)
    print(out)

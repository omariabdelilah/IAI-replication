#!/usr/bin/env python3
"""
03_length_filter_audit.py
=========================
Audit of the fewer-than-three-words exclusion (Section 3.3; second-round
reviewer comment R1.1).

Rebuilds the volume-capped corpus (215,066 reviews) from the raw per-application
scrape files, flags the reviews removed by the length filter, and reports

  (1) removed counts and shares per application,
  (2) the star-rating distribution of removed vs. retained reviews,
  (3) the direction of the resulting bias, and
  (4) a post-stratification sensitivity analysis: retained reviews are
      re-weighted, within each application, so that their star-rating
      distribution matches that of the full volume-capped corpus, and the
      application AQI means, ranking and partition are recomputed.

Requires the raw review text, which is not redistributed (Google Play terms of
service). Its aggregate outputs are deposited in
data/length_filter_audit.json.

    python scripts/03_length_filter_audit.py --raw DIR_WITH_<app>_FINAL.xlsx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from aqi_core import (APP_ORDER, APP_LABEL, HIGHER, LOWER, load, build_components,  # noqa: E402
                      aqi, kendall, partition_preserved)


def read_raw(raw_dir):
    frames = []
    for app in APP_ORDER:
        pk = Path(raw_dir) / f"{app}__Sheet1.pkl"
        if pk.exists():
            d = pd.read_pickle(pk)
        else:
            d = pd.read_excel(Path(raw_dir) / f"{app}_FINAL.xlsx")
        d["app"] = app
        frames.append(d)
    return frames


def capped_corpus(frames):
    out = []
    for d in frames:
        d = d.drop_duplicates("review_id")
        if d["app"].iloc[0] == "opay":
            d = d.sample(n=50000, random_state=42)
        out.append(d)
    c = pd.concat(out, ignore_index=True)
    c["n_words"] = c["text"].fillna("").astype(str).str.split().str.len()
    c["removed"] = c["n_words"] < 3
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data", default="data/extracted_signals.csv")
    ap.add_argument("--out", default="data/length_filter_audit.json")
    a = ap.parse_args()

    c = capped_corpus(read_raw(a.raw))
    R = {"capped_n": len(c), "removed_n": int(c["removed"].sum()),
         "retained_n": int((~c["removed"]).sum()),
         "removed_share": round(100 * c["removed"].mean(), 2)}
    assert R["capped_n"] == 215066 and R["retained_n"] == 124975

    # (1) per application
    per = {}
    for app, g in c.groupby("app"):
        per[app] = dict(capped=len(g), removed=int(g["removed"].sum()),
                        retained=int((~g["removed"]).sum()),
                        removed_share=round(100 * g["removed"].mean(), 1),
                        mean_rating_removed=round(g[g["removed"]]["rating"].mean(), 2),
                        mean_rating_retained=round(g[~g["removed"]]["rating"].mean(), 2),
                        share5_removed=round(100 * (g[g["removed"]]["rating"] == 5).mean(), 1),
                        share5_retained=round(100 * (g[~g["removed"]]["rating"] == 5).mean(), 1),
                        share1_removed=round(100 * (g[g["removed"]]["rating"] == 1).mean(), 1),
                        share1_retained=round(100 * (g[~g["removed"]]["rating"] == 1).mean(), 1))
    R["per_app"] = per
    R["removed_share_range"] = [min(v["removed_share"] for v in per.values()),
                                max(v["removed_share"] for v in per.values())]
    ct = pd.crosstab(c["app"], c["removed"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    R["app_by_removed_chi2"] = dict(chi2=round(chi2, 1), dof=int(dof), p=float(p),
                                    cramers_v=round(np.sqrt(chi2 / (len(c) * (min(ct.shape) - 1))), 3))

    # (2) star distribution
    dist = pd.crosstab(c["removed"], c["rating"], normalize="index") * 100
    R["star_dist_removed"] = {int(k): round(v, 1) for k, v in dist.loc[True].items()}
    R["star_dist_retained"] = {int(k): round(v, 1) for k, v in dist.loc[False].items()}
    R["mean_rating_removed"] = round(c[c["removed"]]["rating"].mean(), 2)
    R["mean_rating_retained"] = round(c[~c["removed"]]["rating"].mean(), 2)
    ct2 = pd.crosstab(c["removed"], c["rating"])
    chi2, p, dof, _ = stats.chi2_contingency(ct2)
    R["star_by_removed_chi2"] = dict(chi2=round(chi2, 1), dof=int(dof), p=float(p),
                                     cramers_v=round(np.sqrt(chi2 / len(c)), 3))
    extreme = c["rating"].isin([1, 5])
    R["extreme_share_removed"] = round(100 * extreme[c["removed"]].mean(), 1)
    R["extreme_share_retained"] = round(100 * extreme[~c["removed"]].mean(), 1)

    # (3)+(4) post-stratification on star rating within application
    df = load(a.data)
    s, i, b = build_components(df)
    df["AQI"] = aqi(s, i, b)
    df["SB"] = aqi(s, i, b, (0.5, 0, 0.2))
    target = c.groupby(["app", "rating"]).size() / c.groupby("app").size()
    sample = df.groupby(["app", "rating"]).size() / df.groupby("app").size()
    wts = (target / sample).rename("w")
    df = df.join(wts, on=["app", "rating"])
    base = df.groupby("app")["AQI"].mean()
    rew = df.groupby("app").apply(lambda g: np.average(g["AQI"], weights=g["w"]))
    base_sb = df.groupby("app")["SB"].mean()
    rew_sb = df.groupby("app").apply(lambda g: np.average(g["SB"], weights=g["w"]))
    R["poststrat"] = {
        app: dict(base=round(base[app], 3), reweighted=round(rew[app], 3),
                  diff=round(rew[app] - base[app], 3))
        for app in APP_ORDER}
    R["poststrat_tau"] = round(kendall(base, rew), 3)
    R["poststrat_partition_preserved"] = bool(partition_preserved(rew))
    R["poststrat_ranking"] = list(rew.sort_values(ascending=False).index)
    R["poststrat_tau_SB"] = round(kendall(base_sb, rew_sb), 3)
    R["poststrat_partition_preserved_SB"] = bool(partition_preserved(rew_sb))
    hw = df[df["app"].isin(HIGHER)]; lw = df[df["app"].isin(LOWER)]
    R["poststrat_group_means"] = [round(np.average(hw["AQI"], weights=hw["w"]), 3),
                                  round(np.average(lw["AQI"], weights=lw["w"]), 3)]
    R["poststrat_gap"] = round(R["poststrat_group_means"][0] - R["poststrat_group_means"][1], 3)
    R["base_gap"] = round(hw["AQI"].mean() - lw["AQI"].mean(), 3)
    # star-rating ranking of applications, full capped vs retained
    rr_full = c.groupby("app")["rating"].mean(); rr_ret = c[~c["removed"]].groupby("app")["rating"].mean()
    R["star_ranking_tau_full_vs_retained"] = round(kendall(rr_full, rr_ret), 3)

    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

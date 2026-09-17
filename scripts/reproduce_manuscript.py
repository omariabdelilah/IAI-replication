#!/usr/bin/env python3
"""
reproduce_manuscript.py
=======================
Regenerates the index-dependent results reported in

    Omari, A. & Satauri, I.
    "Beyond Adoption: An Interpretable, Review-Based Index of FinTech Adoption
     Quality - Evidence from Six African Mobile Finance Applications"
    (second revision)

from the deposited extracted-signal dataset and prints each computed value
next to the value printed in the manuscript. The index is recomputed from the
raw extracted signals (the stored iai_score column is not used).

Second revision
---------------
* The index is renamed the Adoption Quality Index (AQI).
* The baseline barrier count uses only the nine declared barrier categories
  (snake_case spellings normalised). The first-revision baseline, which counted
  every extracted string, is reproduced with --labels all.

Usage
-----
    python scripts/reproduce_manuscript.py
    python scripts/reproduce_manuscript.py --labels all      # first-revision numbers
    python scripts/reproduce_manuscript.py --raw DIR         # + exact-text duplicate row

Scores are rounded to 10 decimal places after aggregation so that binary
floating-point error does not split identical scores (this matters for rank
statistics).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aqi_core import APP_ORDER, APP_LABEL, load  # noqa: E402
from compute_results import compute  # noqa: E402

# value printed in the manuscript (second revision), tolerance
EXPECTED = {
    "r_review": (0.665, 5e-4), "rho_review": (0.599, 5e-4), "r_review_SB": (0.683, 5e-4),
    "higher_mean": (0.812, 5e-4), "lower_mean": (0.600, 5e-4),
    "cohens_d": (0.70, 5e-3), "cliffs_delta": (0.31, 5e-3),
    "share_max_all": (59.5, 0.05), "share_max_SB_all": (64.2, 0.05),
    "dR2_full": (0.014, 5e-4), "nested_F": (1615.0, 0.05),
    "anova_F": (24932, 0.5), "kw_H": (48990, 0.5),
    "corr_SB": (0.772, 5e-4), "corr_IB": (0.345, 5e-4),
    "counterfactual_2021": (0.644, 5e-4), "raw_change_pct": (21.5, 0.05),
    "reliability_share": (13.1, 0.05), "app_r_rating": (0.976, 5e-4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extracted_signals.csv")
    ap.add_argument("--labels", default="declared", choices=["declared", "all"])
    ap.add_argument("--raw", default=None)
    a = ap.parse_args()
    R = compute(load(a.data), a.labels, a.raw)

    print("=" * 78)
    print(f"AQI results  (barrier labels: {a.labels})   n = {R['n']:,}")
    print("=" * 78)
    if a.labels == "declared":
        bad = 0
        for k, (exp, tol) in EXPECTED.items():
            got = R[k]
            ok = abs(got - exp) <= tol
            bad += not ok
            print(f"  {'OK ' if ok else '!! '} {k:<22s} {got:>12}   (manuscript {exp})")
        print(f"\n  {len(EXPECTED) - bad}/{len(EXPECTED)} headline values match.\n")

    print("Table 9 - AQI by application")
    for app in APP_ORDER:
        t = R["table8"][app]
        print(f"  {APP_LABEL[app]:<12s} mean {t['mean']:.3f}  S+B {t['SB']:.3f}  median {t['median']:.2f}"
              f"  at max {t['share_max']:.1f}%  pos {t['pos']:.1f}%  neg {t['neg']:.1f}%  eng {t['eng']:.1f}%")
    print("\nTable 6 - consolidated sensitivity analyses")
    for k, v in R["table6"].items():
        print(f"  {k:<28s} tau {v['tau']:.3f}   r {v['r']:.3f}   partition "
              f"{'preserved' if v['partition'] else 'CHANGED'}   max |d mean| {v['max_mean_diff']:.3f}")
    print("\nFull output: python scripts/compute_results.py --out results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

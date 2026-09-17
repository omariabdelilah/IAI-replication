"""
aqi_core.py
===========
Shared definitions for the Adoption Quality Index (AQI).

The index was called the Inclusive Adoption Index (IAI) in earlier versions
of the manuscript; it was renamed in the second revision because nothing in
its construct definition (Section 3.6) concerns inclusion.

Baseline specification (second revision)
----------------------------------------
Barrier count n_i counts only labels belonging to the nine declared barrier
categories of the extraction schema. Labels that are snake_case spellings of
a declared category (e.g. ``failed_transactions``) are normalised to that
category before counting; all other non-conforming strings are not counted.
The previous baseline, which counted every extracted string, is retained as a
sensitivity analysis ("all extracted labels").
"""

import ast

import numpy as np
import pandas as pd
from scipy import stats

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
APP_ORDER = ["mpesa", "wave", "opay", "pocket_bank", "cih_pay", "mtn_momo"]
APP_LABEL = {
    "mpesa": "M-Pesa", "wave": "Wave", "opay": "OPay",
    "pocket_bank": "Pocket Bank", "cih_pay": "CIH Pay", "mtn_momo": "MTN MoMo",
}
COUNTRY_LABEL = {"ke": "Kenya", "sn": "Senegal", "ng": "Nigeria",
                 "ma": "Morocco", "ug": "Uganda", "gh": "Uganda"}


def parse_list(cell):
    if isinstance(cell, str):
        try:
            value = ast.literal_eval(cell)
            return value if isinstance(value, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def normalise_label(label):
    """snake_case spelling of a declared category -> canonical label."""
    return label.replace("_", " ").strip().lower() if isinstance(label, str) else label


def declared_barriers(lst):
    """Declared-category barrier labels in a review (normalised, de-duplicated)."""
    out = []
    for x in lst:
        y = normalise_label(x)
        if y in DECLARED_BARRIERS and y not in out:
            out.append(y)
    return out


def load(path="data/extracted_signals.csv"):
    df = pd.read_csv(path, low_memory=False)
    df["_barriers_raw"] = df["adoption_barriers"].map(parse_list)
    df["_barriers"] = df["_barriers_raw"].map(declared_barriers)
    df["_cultural"] = df["cultural_factors"].map(parse_list)
    return df


def build_components(df, neutral=0.3, barrier_mode="linear", labels="declared"):
    """Return S, I, B.  labels: 'declared' (baseline) or 'all' (previous baseline)."""
    score_map = dict(SENTIMENT_SCORE, neutral=neutral)
    s = df["sentiment"].map(score_map).astype(float)
    i = (df["inclusion_signal"] != "none").astype(float)
    lists = df["_barriers"] if labels == "declared" else df["_barriers_raw"]
    if barrier_mode == "reliability_x2":
        counts = lists.map(lambda l: sum(2 if x in RELIABILITY_BARRIERS else 1 for x in l))
    else:
        counts = lists.map(len)
    if barrier_mode == "binary":
        b = np.where(counts > 0, 0.5, 1.0)
    elif barrier_mode == "reciprocal":
        b = 1.0 / (1.0 + counts)
    else:
        b = np.maximum(0.0, 1.0 - 0.25 * counts)
    return s, i, pd.Series(np.asarray(b, dtype=float), index=df.index)


def aqi(s, i, b, weights=(ALPHA, BETA, GAMMA)):
    a, be, g = weights
    return ((a * s + be * i + g * b) / (a + be + g)).round(10)


def kendall(base_means, other_means):
    apps = sorted(base_means.index)
    return stats.kendalltau(base_means[apps], other_means[apps]).correlation


def partition_preserved(means):
    return set(means.sort_values(ascending=False).index[:3]) == set(HIGHER)


def cliffs_delta(x, y):
    """Cliff's delta P(X>Y) - P(X<Y) for discrete data, computed via value counts."""
    vals = np.union1d(np.unique(x), np.unique(y))
    cx = pd.Series(x).value_counts().reindex(vals, fill_value=0).values.astype(float)
    cy = pd.Series(y).value_counts().reindex(vals, fill_value=0).values.astype(float)
    cum_y_below = np.concatenate([[0], np.cumsum(cy)[:-1]])
    cum_y_above = cy.sum() - np.cumsum(cy)
    gt = (cx * cum_y_below).sum()
    lt = (cx * cum_y_above).sum()
    return (gt - lt) / (cx.sum() * cy.sum())


def cliffs_delta_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    boots = [cliffs_delta(rng.choice(x, len(x)), rng.choice(y, len(y)))
             for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled

#!/usr/bin/env python3
"""
02_nlp_benchmark.py
===================
Reproduces the NLP baseline comparison of Section 4.9 (Tables 16-18): VADER,
zero-shot mBERT, and the Gemini 2.5 Flash extraction, evaluated against Google
Play star ratings as a three-class proxy ground truth.

Ground truth mapping:  1-2 stars -> negative | 3 -> neutral | 4-5 -> positive.
Mixed-sentiment reviews (n = 13,130; 10.5%) are excluded because the proxy is
three-class; this exclusion is non-random and is discussed in Section 4.9.

REQUIRES THE RAW REVIEW TEXT, which is not redistributed in this repository in
accordance with Google Play's terms of service. mBERT predictions are cached to
`mbert_predictions.csv` so the script can be resumed; that cache contains only
review identifiers and predicted labels and is safe to share.

Usage
-----
    python scripts/02_nlp_benchmark.py --input CORPUS_lang_corrected.xlsx

Expected output on the study corpus (n = 111,823):
    VADER   acc 0.542  F1 0.627  kappa 0.184
    mBERT   acc 0.818  F1 0.823  kappa 0.542
    Gemini  acc 0.832  F1 0.841  kappa 0.586
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

MBERT_CHECKPOINT = "nlptown/bert-base-multilingual-uncased-sentiment"
STAR_TO_CLASS = {0: "negative", 1: "negative", 2: "neutral",
                 3: "positive", 4: "positive"}


def vader_labels(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    out = []
    for text in texts:
        compound = analyser.polarity_scores(text)["compound"]
        out.append("positive" if compound > 0.05
                   else "negative" if compound < -0.05
                   else "neutral")
    return out


def mbert_labels(df, cache="mbert_predictions.csv", batch_size=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    preds = {}
    if os.path.exists(cache):
        cached = pd.read_csv(cache)
        preds = dict(zip(cached["review_id"], cached["mbert"]))

    todo = df[~df["review_id"].isin(preds)].reset_index(drop=True)
    print(f"mBERT: {len(todo):,} reviews to predict "
          f"({len(preds):,} loaded from cache)")
    if not len(todo):
        return df["review_id"].map(preds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MBERT_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MBERT_CHECKPOINT).to(device).eval()

    def flush():
        pd.DataFrame({"review_id": list(preds), "mbert": list(preds.values())}
                     ).to_csv(cache, index=False)

    since = 0
    for start in range(0, len(todo), batch_size):
        chunk = todo.iloc[start:start + batch_size]
        encoded = tokenizer(chunk["text"].astype(str).tolist(), padding=True,
                            truncation=True, max_length=512,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            stars = model(**encoded).logits.argmax(-1).cpu().numpy()
        for rid, star in zip(chunk["review_id"], stars):
            preds[rid] = STAR_TO_CLASS[int(star)]
        since += len(chunk)
        if since >= 2000:
            flush()
            since = 0
    flush()
    return df["review_id"].map(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cache", default="mbert_predictions.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.endswith((".xlsx", ".xls")) \
        else pd.read_csv(args.input, low_memory=False)

    bench = df[df["sentiment"] != "mixed"].copy().reset_index(drop=True)
    bench["truth"] = np.where(bench["rating"] <= 2, "negative",
                              np.where(bench["rating"] == 3, "neutral", "positive"))
    print(f"Benchmark sample: {len(bench):,}  (manuscript 111,823)")

    bench["vader"] = vader_labels(bench["text"].astype(str))
    bench["mbert"] = mbert_labels(bench, cache=args.cache)
    assert bench["mbert"].isna().sum() == 0, "mBERT predictions incomplete - rerun"

    print("\n=== TABLE 16 - overall ===")
    for name, col in [("VADER", "vader"), ("mBERT", "mbert"), ("Gemini", "sentiment")]:
        print(f"  {name:<8s} acc={accuracy_score(bench.truth, bench[col]):.3f}  "
              f"F1={f1_score(bench.truth, bench[col], average='weighted'):.3f}  "
              f"kappa={cohen_kappa_score(bench.truth, bench[col]):.3f}")

    def by(group_col, order=None):
        keys = order or sorted(bench[group_col].unique())
        rows = []
        for key in keys:
            sub = bench[bench[group_col] == key]
            if not len(sub):
                continue
            rows.append({
                group_col: key, "N": len(sub),
                "VADER": round(accuracy_score(sub.truth, sub.vader), 3),
                "mBERT": round(accuracy_score(sub.truth, sub.mbert), 3),
                "Gemini": round(accuracy_score(sub.truth, sub.sentiment), 3),
            })
        return pd.DataFrame(rows)

    print("\n=== TABLE 17 - by language ===")
    lang_col = "lang_corrected" if "lang_corrected" in bench.columns else "lang"
    t17 = by(lang_col, ["ar", "fr", "en"])
    print(t17.to_string(index=False))
    gap = (t17.loc[0, "Gemini"] - t17.loc[0, "mBERT"]) * 100
    print(f"  Arabic gap: {gap:.1f} percentage points  (manuscript 11.3)")

    print("\n=== TABLE 18 - by application ===")
    t18 = by("app")
    print(t18.to_string(index=False))

    t17.to_csv("table17_by_language.csv", index=False)
    t18.to_csv("table18_by_application.csv", index=False)
    print("\nwritten -> table17_by_language.csv, table18_by_application.csv")


if __name__ == "__main__":
    main()

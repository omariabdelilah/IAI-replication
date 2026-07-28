#!/usr/bin/env python3
"""
01_language_rederivation.py
===========================
Re-derives the language label of each review from the review text, replacing the
Google Play store locale under which the review was collected (Section 3.2 of the
manuscript).

Three rules, applied in order:

  1. Arabic script share >= 30% (with at least three Arabic characters) -> "ar".
     Deterministic; no probabilistic inference.
  2. Otherwise, py3langid constrained to {en, fr} with normalised probabilities.
  3. Where identification is unreliable -- fewer than three Latin-script words, or
     a classification probability below 0.90 -- the original store locale is
     retained rather than replaced by an uncertain estimate.

The method applied to each review is recorded in `lang_method`, so the provenance
of every label is auditable:

  arabic_script  | langid | locale_short | locale_lowconf

REQUIRES THE RAW REVIEW TEXT, which is not redistributed in this repository in
accordance with Google Play's terms of service. The resulting labels are
deposited in data/extracted_signals.csv (columns lang_corrected, lang_method).

Usage
-----
    python scripts/01_language_rederivation.py --input CORPUS_ANALYZED.xlsx \
                                               --output CORPUS_lang_corrected.xlsx

Expected output on the study corpus:
    corrected distribution   3,437 ar / 41,006 fr / 80,510 en
    relabelled               4,345 reviews (3.5%)
    store locale retained    7,872 reviews (6.3%)
"""

import argparse
import re

import numpy as np
import pandas as pd
import py3langid

ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")

ARABIC_SHARE_THRESHOLD = 0.30
MIN_ARABIC_CHARS = 3
MIN_LATIN_WORDS = 3
CONFIDENCE_THRESHOLD = 0.90


def rederive(texts, locales):
    identifier = py3langid.langid.LanguageIdentifier.from_pickled_model(
        py3langid.langid.MODEL_FILE, norm_probs=True
    )
    identifier.set_languages(["en", "fr"])

    labels, methods = [], []
    for text, locale in zip(texts, locales):
        n_ar = len(ARABIC.findall(text))
        n_la = len(LATIN.findall(text))
        share = n_ar / (n_ar + n_la) if (n_ar + n_la) else 0.0

        if share >= ARABIC_SHARE_THRESHOLD and n_ar >= MIN_ARABIC_CHARS:
            labels.append("ar")
            methods.append("arabic_script")
            continue

        if len(LATIN_WORD.findall(text)) < MIN_LATIN_WORDS:
            labels.append(locale if locale in ("en", "fr") else "en")
            methods.append("locale_short")
            continue

        lang, prob = identifier.classify(text)
        if prob < CONFIDENCE_THRESHOLD:
            labels.append(locale if locale in ("en", "fr") else lang)
            methods.append("locale_lowconf")
        else:
            labels.append(lang)
            methods.append("langid")

    return labels, methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="corpus with a `text` and a `lang` column")
    ap.add_argument("--output", default="CORPUS_lang_corrected.xlsx")
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.endswith((".xlsx", ".xls")) \
        else pd.read_csv(args.input, low_memory=False)

    df["lang_corrected"], df["lang_method"] = rederive(
        df["text"].astype(str), df["lang"]
    )

    print("\nDeclared vs corrected:")
    print(pd.crosstab(df["lang"], df["lang_corrected"], margins=True))
    print("\nPer application:")
    print(pd.crosstab(df["app"], df["lang_corrected"]))

    relabelled = int((df["lang"] != df["lang_corrected"]).sum())
    retained = int(df["lang_method"].isin(["locale_short", "locale_lowconf"]).sum())
    print(f"\nrelabelled           {relabelled:,} ({100 * relabelled / len(df):.1f}%)")
    print(f"store locale kept    {retained:,} ({100 * retained / len(df):.1f}%)")

    if args.output.endswith((".xlsx", ".xls")):
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)
    print(f"\nwritten -> {args.output}")


if __name__ == "__main__":
    main()

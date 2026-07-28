# Inclusive Adoption Index (IAI) — replication materials

Data and code accompanying:

> Omari, A. & Satauri, I. **Beyond Adoption: An Interpretable, Review-Based Index
> of FinTech Adoption Quality — Evidence from Six African Mobile Finance
> Applications.** *Scientific African* (under review).

The Inclusive Adoption Index (IAI) measures the **quality** of FinTech adoption —
rather than its extent — directly from user-generated app-store reviews. It
combines three components: affective evaluation (sentiment), behavioural
continuance (engagement), and the absence of experienced friction (barriers).

---

## What is here

```
data/
  extracted_signals.csv          124,953 rows — the analytical sample
scripts/
  reproduce_manuscript.py        regenerates every number in the paper
  01_language_rederivation.py    text-based language labelling (Section 3.2)
  02_nlp_benchmark.py            VADER / mBERT / Gemini comparison (Section 4.9)
requirements.txt
```

## Review texts are not redistributed

`data/extracted_signals.csv` contains **review identifiers and extracted
categorical signals only**. The review text is deliberately excluded, in
accordance with the Google Play Store terms of service. Reviews can be
re-retrieved from their identifiers by anyone with access to the platform.

Everything the paper *concludes* is reproducible from this file. The two
text-dependent stages — language re-derivation and the NLP benchmark — are
provided as scripts so the procedure is auditable, and their **outputs** are
deposited here (the `lang_corrected` and `lang_method` columns).

---

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/reproduce_manuscript.py
```

Runtime is under a minute. The script recomputes the IAI from the raw extracted
signals rather than reading the stored `iai_score` column, so the index formula
itself is verified. It prints the manuscript value next to each computed value.

Covered: sample cascade (Table 5) · Table 8 · sentiment distribution (§4.2) ·
Table 11 with effect sizes · Table 12 · Table 13 · within-stratum evidence
(§4.7) · group separation and Cohen's *d* (§4.10) · Table 9 decomposition ·
Figure 6 barriers · Figure 7 cultural factors · engagement (§4.5) ·
non-conforming labels (§3.4) · empirical weights (§3.6) · Table 14 ablation ·
the full Table 6 sensitivity battery.

---

## Data dictionary

| Column | Type | Description |
|---|---|---|
| `review_id` | str | Google Play review identifier |
| `app` | str | `mpesa`, `wave`, `opay`, `cih_pay`, `pocket_bank`, `mtn_momo` |
| `country` | str | Primary operational market (ISO-2). **Assigned per application, not per user** — Google Play exposes no user-level geography |
| `year`, `date` | int, str | Review timestamp, 2019–2024 |
| `rating` | int | Star rating, 1–5. Used as the external validation criterion |
| `likes` | int | Helpfulness votes |
| `lang` | str | Store locale of the collection stream (`en`/`fr`/`ar`) — **not** a reliable language indicator |
| `lang_corrected` | str | Language re-derived from the review text (Section 3.2) |
| `lang_method` | str | How the label was obtained: `arabic_script`, `langid`, `locale_short`, `locale_lowconf` |
| `sentiment` | str | `positive` / `mixed` / `neutral` / `negative` |
| `adoption_drivers` | str | JSON-style list, closed taxonomy of 9 categories |
| `adoption_barriers` | str | JSON-style list, closed taxonomy of 9 categories |
| `cultural_factors` | str | JSON-style list, closed taxonomy of 6 categories |
| `inclusion_signal` | str | `strong` / `moderate` / `weak` / `none` |
| `n_barriers` | int | Count of barrier mentions, i.e. *n<sub>i</sub>* in Eq. (3) |
| `iai_score` | float | Per-review IAI under the baseline specification |

### Known data caveat

Schema validation at extraction time was structural, not semantic: membership of
the declared closed vocabularies was not enforced at parse time. As a result
**5,614 label mentions across 31 strings outside the nine declared barrier
categories** appear in `adoption_barriers`, affecting 5,309 reviews (4.25%).
This is a defect in schema enforcement, disclosed in Section 3.4 of the paper
and reproduced here unaltered rather than silently cleaned.

Its direction was checked: the affected reviews are predominantly negative (mean
star rating 2.20 against 4.07 for the corpus). The alternative specification
restricted to the nine declared categories is reported as a robustness check —
application means differ by at most 0.006, the ranking is identical
(Kendall's *τ* = 1.000), and the review-level correlation moves from
*r* = 0.667 to 0.665. `reproduce_manuscript.py` computes both.

---

## The index

For each review *i*:

```
IAI_i = α·S_i + β·I_i + γ·B_i        α = 0.50, β = 0.30, γ = 0.20
B_i   = max(0, 1 − 0.25 · n_i)
I_i   = 1 if any engagement signal is present, else 0
S_i   = 1.00 positive | 0.50 mixed | 0.30 neutral | 0.00 negative
```

Weights are theoretically motivated rather than fitted. Empirically estimated
(OLS, normalised) weights are 0.488 / 0.087 / 0.425; application rankings are
invariant between the two schemes (*τ* = 1.000). An `S + B` variant with
engagement removed is reported throughout as a declared alternative
specification — every substantive conclusion holds under both.

---

## Extraction

Signals were extracted with **Gemini 2.5 Flash** via the Google Generative AI
API on **25 February 2026**: greedy decoding (temperature 0.0, top-*p* 1.0),
structured JSON array output, 20 reviews per request, 4 concurrent workers, in
fixed corpus order. Star ratings were withheld from the prompt to prevent label
leakage into the benchmark. The full prompt is in Section 3.4 of the paper.

Greedy decoding maximises run-to-run consistency but does not guarantee
bit-for-bit reproducibility of batched API inference across inference dates or
provider-side infrastructure changes. This is why the extracted signals — not
just the extraction code — are deposited.

Inter-rater agreement was assessed on a stratified sample of 300 reviews against
an independent second rater (Claude Opus 4.7, 29 April 2026): *κ* = 0.738 for
sentiment, mean Jaccard 0.765 for barriers, *κ* = 0.462 for binary engagement.

---

## What this dataset is not

The corpus captures the experience of **Google Play reviewers** — a self-selected
population motivated to publicly rate an application — not of the full user base,
and not of the underlying population of FinTech adopters. The IAI is best read as
an index of *reviewed* adoption experience.

Country-level aggregates are arithmetically determined by application-level
scores: the design assigns essentially one application per country, so country is
not statistically separable from application here. Cross-country claims are not
supported by this data.

---

## Citation

```bibtex
@article{omari2026iai,
  author  = {Omari, Abdelilah and Satauri, Imane},
  title   = {Beyond Adoption: An Interpretable, Review-Based Index of FinTech
             Adoption Quality --- Evidence from Six African Mobile Finance
             Applications},
  journal = {Scientific African},
  year    = {2026},
  note    = {Under review}
}
```

## Licence

Code: MIT. Data (extracted signals): CC BY 4.0. Review identifiers refer to
third-party content on the Google Play Store and remain subject to that
platform's terms.

## Contact

Abdelilah Omari — abdelilah.omari@usmba.ac.ma
L3IA Laboratory, Faculty of Sciences Dhar El Mahraz,
Sidi Mohamed Ben Abdellah University, Fez, Morocco

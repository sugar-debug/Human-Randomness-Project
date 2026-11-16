# Are Humans Actually Random?

A statistical investigation of human-generated randomness, plus a small ML
classifier and live web demo that guesses whether a sequence came from a person
or a computer.

Foundation of Data Science final project · MNIT Jaipur
Team: [Team member names]

## The question

Can people generate sequences that behave like a random process? We looked at
three things:

- **Frequency** – are digits and heads/tails used in the right proportions?
- **Repetition** – do repeats, streaks and runs appear as often as chance predicts?
- **Dependence** – is each choice independent of the one before it?

Extension: can a classifier tell a human sequence from a computer one?

## Data

64 students typed, without any random generator: 20 digits (0–9), 20 coin
flips (H/T), one number from 1 to 100, and answered a short survey. The
computer reference is one fair random pair per participant, with the same
lengths (seed 42), simulated 10,000 times for the tests. Five near-constant
responses were set aside by a rule fixed in advance; every test is reported on
all 64 and on the cleaned 59.

Names were replaced by IDs (`P001`…) before analysis.

## Headline results

| Finding | Result |
|---|---|
| Digit frequency vs uniform | χ² = 30.3, p < 0.001 (0 and 1 used far less than 128 expected) |
| Digit entropy | Humans 3.04 vs computer 2.95 bits, p < 0.001 |
| No repeated neighbouring digit | 53% of humans vs 14% of computers |
| Coin balance | 52% heads, p = 0.14, no heads bias (after removing 4 all-heads sequences) |
| Coin runs | Humans 11.5 vs computer 10.5, p ≈ 0.001: people switch too often |
| Classifier (random forest) | 78% accuracy, AUC 0.87 (chance = 50%) |
| Classifier (logistic regression, used in the demo) | ~70–76% accuracy depending on the reference set |

People are too even, too balanced, and avoid repeats. Digit features carry most
of the signal (~73% alone); coin flips alone reach only 60–65%.

Full write-up: [`docs/slides.pdf`](docs/slides.pdf).

## Repository layout

```
.
├── app/index.html               # Self-contained web demo (no server needed)
├── data/
│   ├── survey_anonymized.csv    # 64 responses, names replaced by IDs
│   ├── processed/               # per-sequence metrics for humans and computer
│   └── raw/                     # LOCAL ONLY, gitignored (contains names)
├── results/
│   ├── comparison_results_v2.csv  # human vs computer, per metric
│   └── classifier_results.csv     # model / feature-set comparison
├── src/
│   ├── features.py              # the 12 features (mirrors the JS in the demo)
│   ├── simulate_computer.py     # computer reference sequences
│   ├── train_export.py          # train logistic regression, export weights
│   └── anonymize.py             # strip names from raw survey export
├── tests/test_features.py
├── notebooks/                   # put your analysis notebook(s) here
└── docs/slides.pdf
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                  # check the feature code
python src/train_export.py              # retrain + print CV accuracy
```

Open `app/index.html` in any browser to try the demo: type 20 digits and 20
flips (or click *Generate computer sample*) and see which of the 12 features
gave you away.

## The web demo

A logistic regression on 12 features (digit entropy, runs, chi-square,
autocorrelation, adjacent repeat rate, steps of size 1, distinct digits, coin
entropy, coin runs, coin imbalance, switch rate, longest coin run). The fitted
weights, scaling values and feature ranges are stored as plain numbers in the
`M` constant inside `app/index.html`, and the page recomputes the same features
in the browser. Logistic regression is used because its weights run directly in
a browser; the random forest stays in the notebook.

To host it free: enable GitHub Pages on the `main` branch, `/app` folder
(rename or copy `index.html` as needed).

## Limits

- One sample of 64 student volunteers; results describe these students, not people in general.
- Only 20 items per sequence, so the demo accepts exactly 20 digits and 20 flips.
- A single verdict is a lean, not proof.
- Pooled tests assume independent flips.

## Future work

Longer sequences, a wider sample, pairs/triples of digits and time-series
models, does feedback reduce the bias, and running the random forest in the
browser.

## License

MIT, see [LICENSE](LICENSE).

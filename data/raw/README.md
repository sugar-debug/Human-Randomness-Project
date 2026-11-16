# data/raw

Holds the original survey export that still contains student names.
Everything in this folder except this file is gitignored and must never be
pushed. Run `python src/anonymize.py <raw.csv> data/survey_anonymized.csv` to
produce the shareable version.

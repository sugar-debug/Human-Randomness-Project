"""Replace names with participant IDs before the data goes anywhere public.

Usage:
    python src/anonymize.py data/raw/cleaned_survey_data_with_names.csv data/survey_anonymized.csv
"""
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
if "Name" in df.columns:
    df = df.drop(columns=["Name"])
if "participant_id" not in df.columns:
    df.insert(0, "participant_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
df.to_csv(dst, index=False)
print(f"wrote {dst} ({len(df)} rows)")

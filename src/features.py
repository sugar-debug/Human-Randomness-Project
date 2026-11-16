"""The 12 sequence features used by the classifier.

This mirrors the JavaScript in app/index.html exactly, so the Python model and
the web demo score the same input identically.
"""
from __future__ import annotations

import math
import re

import numpy as np

FEATURE_COLS = [
    "digit_entropy", "digit_runs", "digit_chi2", "digit_autocorr",
    "adj_repeat_rate", "diff1_rate", "distinct_digits",
    "binary_entropy", "binary_runs", "binary_balance_dev",
    "switch_rate", "longest_run",
]


def parse_digits(s: str) -> list[int]:
    """Extract every 0-9 character; spaces and commas are ignored."""
    return [int(c) for c in re.findall(r"\d", str(s))]


def parse_flips(s: str) -> list[int]:
    """Heads -> 1, tails -> 0. Anything other than H/T is ignored."""
    return [1 if c == "H" else 0 for c in re.findall(r"[HT]", str(s).upper())]


def entropy(counts) -> float:
    n = sum(counts)
    return -sum((k / n) * math.log2(k / n) for k in counts if k > 0)


def n_runs(x) -> int:
    return 1 + sum(1 for i in range(1, len(x)) if x[i] != x[i - 1])


def longest_run(x) -> int:
    best = cur = 1
    for i in range(1, len(x)):
        cur = cur + 1 if x[i] == x[i - 1] else 1
        best = max(best, cur)
    return best


def autocorr(x) -> float:
    """Lag-1 Pearson correlation; 0 if either side is constant."""
    a, b = np.asarray(x[:-1], float), np.asarray(x[1:], float)
    da, db = a - a.mean(), b - b.mean()
    sa, sb = (da ** 2).sum(), (db ** 2).sum()
    if sa == 0 or sb == 0:
        return 0.0
    return float((da * db).sum() / math.sqrt(sa * sb))


def sequence_features(digits: list[int], flips: list[int]) -> dict[str, float]:
    n = len(digits)
    counts = [0] * 10
    for d in digits:
        counts[d] += 1
    expected = n / 10
    chi2 = sum((k - expected) ** 2 / expected for k in counts)
    repeats = sum(1 for i in range(1, n) if digits[i] == digits[i - 1])
    steps_of_1 = sum(1 for i in range(1, n) if abs(digits[i] - digits[i - 1]) == 1)

    nb = len(flips)
    heads = sum(flips)
    switches = sum(1 for i in range(1, nb) if flips[i] != flips[i - 1])

    return {
        "digit_entropy": entropy(counts),
        "digit_runs": n_runs(digits),
        "digit_chi2": chi2,
        "digit_autocorr": autocorr(digits),
        "adj_repeat_rate": repeats / (n - 1),
        "diff1_rate": steps_of_1 / (n - 1),
        "distinct_digits": sum(1 for k in counts if k > 0),
        "binary_entropy": entropy([nb - heads, heads]),
        "binary_runs": n_runs(flips),
        "binary_balance_dev": abs(heads / nb - 0.5),
        "switch_rate": switches / (nb - 1),
        "longest_run": longest_run(flips),
    }

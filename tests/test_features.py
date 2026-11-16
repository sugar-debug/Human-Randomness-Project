import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import parse_digits, parse_flips, sequence_features  # noqa: E402


def test_parsing_ignores_separators():
    assert parse_digits("1, 2 3,4") == [1, 2, 3, 4]
    assert parse_flips("h t, H") == [1, 0, 1]


def test_perfectly_even_digits():
    f = sequence_features(list(range(10)) * 2, [1, 0] * 10)
    assert math.isclose(f["digit_entropy"], math.log2(10))
    assert f["digit_chi2"] == 0
    assert f["distinct_digits"] == 10
    assert f["adj_repeat_rate"] == 0
    assert f["switch_rate"] == 1
    assert f["longest_run"] == 1


def test_constant_sequence():
    f = sequence_features([5] * 20, [1] * 20)
    assert f["digit_entropy"] == 0
    assert f["digit_runs"] == 1
    assert f["digit_autocorr"] == 0
    assert f["binary_balance_dev"] == 0.5
    assert f["longest_run"] == 20


def test_demo_example_matches_app():
    # The human-style example preloaded in app/index.html
    d = parse_digits("3 7 1 9 4 0 8 2 6 5 7 3 1 8 4 9 0 2 6 5")
    b = parse_flips("H T H H T H T T H T H T T H H T H T H T")
    f = sequence_features(d, b)
    assert f["digit_entropy"] > 3.3   # every digit exactly twice
    assert f["binary_runs"] == 16

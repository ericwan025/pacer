"""Calibration lowers ECE on a deliberately miscalibrated model."""

import numpy as np

from pacer.eval.metrics import ece
from pacer.models.calibrate import (
    IsotonicCalibrator,
    PlattCalibrator,
    fit_best_calibrator,
)


def _miscalibrated(n=20000, seed=0):
    """True prob p_true; model reports an over-confident, squashed version so it
    ranks fine (monotone) but is miscalibrated."""
    rng = np.random.default_rng(seed)
    p_true = rng.beta(1.2, 20, size=n)  # skewed toward 0, like real CTR
    y = (rng.random(n) < p_true).astype(float)
    # monotone distortion: push predictions toward 0.5-ish, add gain
    p_model = np.clip(p_true**0.6 * 1.8, 1e-4, 0.999)
    return p_model, y, p_true


def test_platt_reduces_ece():
    p, y, _ = _miscalibrated(seed=1)
    cal = PlattCalibrator().fit(p, y)
    assert ece(y, cal.predict(p)) < ece(y, p)


def test_isotonic_reduces_ece():
    p, y, _ = _miscalibrated(seed=2)
    cal = IsotonicCalibrator().fit(p, y)
    assert ece(y, cal.predict(p)) < ece(y, p)


def test_isotonic_is_monotone():
    p, y, _ = _miscalibrated(seed=3)
    cal = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0, 1, 100)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= -1e-9)


def test_fit_best_picks_lower_ece_and_reports():
    p, y, _ = _miscalibrated(seed=4)
    cal, report = fit_best_calibrator(p, y)
    assert report["chosen"] in ("platt", "isotonic")
    chosen_ece = min(report["ece_platt_val"], report["ece_isotonic_val"])
    assert chosen_ece <= report["ece_raw_val"]
    # calibrated predictions are valid probabilities
    out = cal.predict(p)
    assert out.min() >= 0 and out.max() <= 1

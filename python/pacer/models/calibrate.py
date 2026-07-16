"""Post-hoc calibration of pCTR.

Because  bid = pCTR * value_per_click * pacing_multiplier , an uncalibrated pCTR
does not merely misrank — it directly scales every bid and corrupts the pacing
controller's input. AUC is invariant to monotone transforms, so a model can rank
well and still be badly miscalibrated. We fit the calibrator on the VALIDATION
split and report ECE before/after on TEST.

Two calibrators:
* Platt scaling: fit sigmoid(a * logit + b). Parametric, robust, monotone.
* Isotonic regression: free-form monotone step function. More flexible, can
  overfit small val sets.
"""

from __future__ import annotations

import numpy as np


def _to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic regression on the model logit: sigmoid(a*z + b)."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        z = _to_logit(np.asarray(p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(z, np.asarray(y))
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = _to_logit(np.asarray(p))
        return 1 / (1 + np.exp(-(self.a * z + self.b)))

    def to_dict(self) -> dict:
        return {"kind": "platt", "a": self.a, "b": self.b}


class IsotonicCalibrator:
    """Monotone piecewise-constant map from pCTR to calibrated prob."""

    def __init__(self):
        self._x = None  # knot x (predicted)
        self._y = None  # knot y (calibrated)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.asarray(p), np.asarray(y))
        # store the fitted thresholds for serialization / manual predict
        self._iso = iso
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(p))

    def to_dict(self) -> dict:
        # export the step function so Go/other tools can apply it without sklearn
        return {
            "kind": "isotonic",
            "x": list(map(float, self._iso.X_thresholds_)),
            "y": list(map(float, self._iso.y_thresholds_)),
        }


def fit_best_calibrator(
    p_val: np.ndarray, y_val: np.ndarray, n_bins: int = 20
) -> tuple[object, dict]:
    """Fit both, pick the one with lower ECE on the val split. Returns
    (calibrator, report)."""
    from pacer.eval.metrics import ece

    platt = PlattCalibrator().fit(p_val, y_val)
    iso = IsotonicCalibrator().fit(p_val, y_val)

    ece_raw = ece(y_val, p_val, n_bins)
    ece_platt = ece(y_val, platt.predict(p_val), n_bins)
    ece_iso = ece(y_val, iso.predict(p_val), n_bins)

    if ece_iso <= ece_platt:
        best, name = iso, "isotonic"
    else:
        best, name = platt, "platt"

    report = {
        "ece_raw_val": ece_raw,
        "ece_platt_val": ece_platt,
        "ece_isotonic_val": ece_iso,
        "chosen": name,
    }
    return best, report

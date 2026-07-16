"""Reliability plot renders to a file on synthetic data."""

import numpy as np

from pacer.eval.plots import plot_reliability


def test_reliability_plot_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.beta(2, 20, 5000)          # skewed, CTR-like
    y = (rng.random(5000) < p).astype(float)
    p_post = np.clip(p * 1.1, 0, 1)    # a fake "calibrated" curve
    out = tmp_path / "rel.png"
    path = plot_reliability(y, p, str(out), p_post=p_post)
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)

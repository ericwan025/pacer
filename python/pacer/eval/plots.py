"""Plotting helpers that don't belong to the sim harness.

Kept import-light: matplotlib is imported inside functions so importing this
module never requires it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pacer.eval.metrics import ece, reliability_curve


def plot_reliability(
    y: np.ndarray,
    p_pre: np.ndarray,
    path: str,
    p_post: np.ndarray | None = None,
    n_bins: int = 20,
) -> str:
    """Reliability diagram: predicted probability vs empirical CTR per equal-mass
    bin, with the y=x diagonal. Optionally overlays a post-calibration curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

    mp, mt, _ = reliability_curve(y, p_pre, n_bins)
    ax.plot(mp, mt, "o-", ms=4, label=f"pre-cal (ECE={ece(y, p_pre, n_bins):.4f})")

    if p_post is not None:
        mp2, mt2, _ = reliability_curve(y, p_post, n_bins)
        ax.plot(mp2, mt2, "s-", ms=4, label=f"post-cal (ECE={ece(y, p_post, n_bins):.4f})")

    lim = max(float(np.max(p_pre)), float(mt.max()) if len(mt) else 0.0, 0.05)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical CTR")
    ax.set_title("Reliability diagram (equal-mass bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

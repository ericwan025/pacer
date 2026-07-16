"""Export a trained model to ONNX for the Go serving path.

We export the BASE network with a final sigmoid, so ONNX emits pCTR directly.
Calibration (isotonic/Platt) is a cheap monotone map serialized to JSON and
applied identically by Python and Go — same pattern as the feature transform.
This keeps step-function calibrators out of a fragile ONNX graph while still
guaranteeing the served, calibrated pCTR matches training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ProbModel(nn.Module):
    """Wrap a base logit model so the graph outputs probability."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base(x))


def export_onnx(
    base: nn.Module,
    n_fields: int,
    path: str,
    opset: int = 17,
) -> str:
    model = ProbModel(base).eval()
    dummy = torch.zeros((2, n_fields), dtype=torch.long)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["features"],
        output_names=["pctr"],
        dynamic_axes={"features": {0: "batch"}, "pctr": {0: "batch"}},
        opset_version=opset,
    )
    return path


def export_calibrator(calibrator, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibrator.to_dict(), f)
    return path


def onnx_predict(path: str, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(["pctr"], {"features": x.astype(np.int64)})[0]
    return out.reshape(-1)

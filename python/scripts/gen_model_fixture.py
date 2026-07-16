"""Generate the Go<->Python ONNX parity fixture.

Builds a small DeepFM with fixed random weights, exports it to ONNX, and writes:
  - model.onnx    : the graph Go loads
  - inputs.json   : int64 feature rows
  - expected.json : pCTR from Python onnxruntime (the ground truth Go must match)

Run: python -m scripts.gen_model_fixture
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pacer.models.deepfm import DeepFM
from pacer.models.export import export_onnx, onnx_predict

OUT = Path("../go/internal/model/testdata")
CARDS = [7, 11, 5, 9, 13, 4, 6, 8]


def main() -> None:
    torch.manual_seed(0)
    base = DeepFM(CARDS, embed_dim=16, mlp_dims=(64, 64), dropout=0.2).eval()

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "model.onnx"
    export_onnx(base, n_fields=len(CARDS), path=str(model_path))

    rng = np.random.default_rng(0)
    x = np.stack([rng.integers(0, c, 256) for c in CARDS], axis=1).astype(np.int64)
    pctr = onnx_predict(str(model_path), x)

    (OUT / "model_inputs.json").write_text(json.dumps(x.tolist()))
    (OUT / "model_expected.json").write_text(json.dumps(pctr.tolist()))
    print(f"wrote model fixture: {len(x)} rows, {len(CARDS)} fields, model -> {model_path}")


if __name__ == "__main__":
    main()

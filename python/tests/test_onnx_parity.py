"""ONNX parity: PyTorch and onnxruntime must agree within 1e-5 on 1000 rows.

If they don't, the Go serving path silently computes a different pCTR and every
downstream bid/pacing number is fiction.
"""

import numpy as np
import torch

from pacer.models.deepfm import DeepFM
from pacer.models.export import ProbModel, export_onnx, onnx_predict


def test_onnx_matches_pytorch_1000_rows(tmp_path):
    cards = [7, 11, 5, 9, 13, 4]
    base = DeepFM(cards, embed_dim=16, mlp_dims=(64, 64), dropout=0.2)
    base.eval()

    rng = np.random.default_rng(0)
    x = np.stack([rng.integers(0, c, 1000) for c in cards], axis=1).astype(np.int64)

    with torch.no_grad():
        torch_p = torch.sigmoid(base(torch.from_numpy(x))).numpy()
    # sanity: ProbModel gives the same
    with torch.no_grad():
        assert np.allclose(ProbModel(base)(torch.from_numpy(x)).numpy(), torch_p, atol=1e-6)

    path = str(tmp_path / "model.onnx")
    export_onnx(base, n_fields=len(cards), path=path)
    onnx_p = onnx_predict(path, x)

    max_diff = float(np.max(np.abs(torch_p - onnx_p)))
    assert max_diff < 1e-5, f"max abs diff {max_diff}"


def test_onnx_dynamic_batch(tmp_path):
    cards = [5, 5, 5]
    base = DeepFM(cards, embed_dim=8, mlp_dims=(16,))
    base.eval()
    path = str(tmp_path / "m.onnx")
    export_onnx(base, n_fields=3, path=path)
    for n in (1, 33, 257):
        x = np.random.default_rng(n).integers(0, 5, (n, 3)).astype(np.int64)
        assert onnx_predict(path, x).shape == (n,)

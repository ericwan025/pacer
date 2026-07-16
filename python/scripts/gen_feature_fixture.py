"""Generate the Go<->Python feature-transform parity fixture.

Fits a small FeatureTransform on synthetic raw rows and writes:
  - transform.json : the artifact Go loads
  - rows.json      : raw string rows the Go test feeds in
  - expected.json  : the encoded vectors (field_order) Python produced

Run: python -m scripts.gen_feature_fixture
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pacer.data.features import FeatureConfig, FeatureTransform

OUT = Path("../go/internal/model/testdata")

RAW_COLS = [
    "hour", "C1", "banner_pos", "site_category", "app_category", "device_type",
    "device_conn_type", "device_id", "device_ip", "site_id", "site_domain",
    "app_id", "app_domain", "device_model",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]


def _rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "hour": 14102100 + (i % 72),  # spans several days/hours
            "click": i % 3 == 0,
            "C1": str(1000 + i % 4),
            "banner_pos": str(i % 3),
            "site_category": ["a", "b", "c"][i % 3],
            "app_category": ["x", "y"][i % 2],
            "device_type": str(i % 2),
            "device_conn_type": str(i % 4),
            "device_id": f"dev{i % 7}",
            "device_ip": f"ip{i % 50}",
            "site_id": f"s{i % 11}",
            "site_domain": f"sd{i % 9}",
            "app_id": f"a{i % 13}",
            "app_domain": f"ad{i % 6}",
            "device_model": f"m{i % 17}",
            "C14": str(100 + i % 20),
            "C15": str(320),
            "C16": str(50),
            "C17": str(i % 5),
            "C18": str(i % 4),
            "C19": str(35),
            "C20": str(-1),
            "C21": str(79),
        })
    return rows


def main() -> None:
    cfg = FeatureConfig(min_count=2)
    cfg.hash_fields = {k: 4096 for k in cfg.hash_fields}  # small buckets, still exercised
    train = pl.DataFrame(_rows(300))
    t = FeatureTransform(cfg).fit(train)

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_json(str(OUT / "transform.json"))

    sample = _rows(40)
    df = pl.DataFrame(sample)
    encoded = t.transform(df)
    field_order = t.field_order()
    expected = [[int(encoded[c][i]) for c in field_order] for i in range(len(sample))]

    # raw rows as string maps for Go (drop the label; keep hour as string)
    raw_rows = [{c: str(r[c]) for c in RAW_COLS} for r in sample]

    (OUT / "rows.json").write_text(json.dumps(raw_rows))
    (OUT / "expected.json").write_text(json.dumps(expected))
    (OUT / "field_order.json").write_text(json.dumps(field_order))

    # also export a DeepFM whose cardinalities MATCH this transform, so the Go
    # end-to-end integration test can go raw row -> transform -> model.
    import torch

    from pacer.models.deepfm import DeepFM
    from pacer.models.export import export_onnx

    torch.manual_seed(0)
    cards = [t.cardinalities()[f] for f in field_order]
    dfm = DeepFM(cards, embed_dim=8, mlp_dims=(32, 32), dropout=0.1).eval()
    export_onnx(dfm, n_fields=len(cards), path=str(OUT / "integration_model.onnx"))

    print(f"wrote fixture to {OUT}/ ({len(sample)} rows, {len(field_order)} fields)")


if __name__ == "__main__":
    main()

"""Phase 2 entrypoint: train LR + DeepFM, calibrate, export ONNX, print the table.

Every number printed here comes from the real Avazu splits. Run `make data`
first. Nothing is fabricated; missing splits abort with a clear message.

Usage: python -m pacer.models.train_ctr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from pacer.data.download import SPLIT_DIR
from pacer.data.features import FeatureConfig, FeatureTransform
from pacer.data.loader import _to_arrays
from pacer.eval.metrics import auc, ece, logloss
from pacer.models.calibrate import fit_best_calibrator
from pacer.models.deepfm import DeepFM
from pacer.models.export import export_calibrator, export_onnx
from pacer.models.lr import HashedLogReg
from pacer.models.train import TrainConfig, fit, predict_proba

ARTIFACT_DIR = Path("artifacts")


def _load_split(name: str, transform: FeatureTransform):
    df = pl.read_parquet(SPLIT_DIR / f"{name}.parquet")
    tdf = transform.transform(df)
    return _to_arrays(tdf, transform.field_order())


def _require_splits():
    missing = [
        str(SPLIT_DIR / f"{s}.parquet")
        for s in ("train", "val", "test")
        if not (SPLIT_DIR / f"{s}.parquet").exists()
    ]
    if missing:
        sys.exit(f"ERROR: missing splits {missing}. Run `make data` first.")


def main() -> None:
    _require_splits()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fit transform on TRAIN, persist for Go.
    train_df = pl.read_parquet(SPLIT_DIR / "train.parquet")
    transform = FeatureTransform(FeatureConfig()).fit(train_df)
    transform.to_json(str(ARTIFACT_DIR / "feature_transform.json"))
    cards = [transform.cardinalities()[f] for f in transform.field_order()]

    xtr, ytr = _to_arrays(transform.transform(train_df), transform.field_order())
    xva, yva = _load_split("val", transform)
    xte, yte = _load_split("test", transform)
    print(f"train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}")
    print(f"base CTR  train={ytr.mean():.4f} val={yva.mean():.4f} test={yte.mean():.4f}")

    results = {}

    # 2a. LR baseline
    print("\n== LR ==")
    lr = HashedLogReg(cards)
    fit(lr, (xtr, ytr), (xva, yva), TrainConfig(lr=0.05, max_epochs=15, patience=2))
    results["LR"] = _eval(lr, xte, yte)

    # 2b. DeepFM
    print("\n== DeepFM ==")
    dfm = DeepFM(cards, embed_dim=16, mlp_dims=(400, 400, 400), dropout=0.2)
    fit(dfm, (xtr, ytr), (xva, yva), TrainConfig(lr=1e-3, max_epochs=20, patience=2))
    results["DeepFM"] = _eval(dfm, xte, yte)

    # 2c. Calibration (fit on VAL, report on TEST) for DeepFM
    print("\n== Calibration (DeepFM) ==")
    p_val = predict_proba(dfm, xva)
    p_test = predict_proba(dfm, xte)
    calibrator, cal_report = fit_best_calibrator(p_val, yva)
    p_test_cal = calibrator.predict(p_test)
    ece_pre = ece(yte, p_test)
    ece_post = ece(yte, p_test_cal)
    results["DeepFM"]["ece_post"] = ece_post
    print(f"  chosen={cal_report['chosen']}  ECE pre={ece_pre:.4f} post={ece_post:.4f}")
    print(f"  avg predicted CTR pre={p_test.mean():.4f} post={p_test_cal.mean():.4f} "
          f"actual={yte.mean():.4f}")

    export_calibrator(calibrator, str(ARTIFACT_DIR / "calibrator.json"))

    # reliability diagram (Phase 2 deliverable)
    from pacer.eval.plots import plot_reliability

    plot_reliability(yte, p_test, "../README_assets/reliability.png", p_post=p_test_cal)

    # 2d. Export DeepFM to ONNX
    export_onnx(dfm, n_fields=len(cards), path=str(ARTIFACT_DIR / "model.onnx"))
    print(f"\nexported artifacts to {ARTIFACT_DIR}/")

    # results table
    print("\n== results table ==")
    print(f"{'model':<8} {'AUC':>7} {'logloss':>9} {'ECE_pre':>9} {'ECE_post':>9}")
    for name, r in results.items():
        print(f"{name:<8} {r['auc']:>7.4f} {r['logloss']:>9.4f} "
              f"{r['ece_pre']:>9.4f} {r.get('ece_post', float('nan')):>9.4f}")

    (ARTIFACT_DIR / "phase2_results.json").write_text(
        json.dumps({"results": results, "calibration": cal_report}, indent=2)
    )


def _eval(model, xte, yte) -> dict:
    p = predict_proba(model, xte)
    r = {
        "auc": auc(yte, p),
        "logloss": logloss(yte, p),
        "ece_pre": ece(yte, p),
    }
    print(f"  test AUC={r['auc']:.4f} logloss={r['logloss']:.4f} ECE={r['ece_pre']:.4f}")
    return r


if __name__ == "__main__":
    main()

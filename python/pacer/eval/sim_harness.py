"""Phase 6 evaluation harness — `make sim`.

Runs all five strategies in smooth and bursty traffic at identical budgets, same
seed, and reports every pacing + delivery metric. Produces the README plots.

Data source:
* REAL (default): loads the test split + the exported model artifacts (feature
  transform, ONNX model, calibrator) and computes calibrated pCTR per impression.
  These numbers are the ones allowed in the README.
* SYNTHETIC (--synthetic): a clearly-labelled diurnal synthetic dataset so the
  harness runs end-to-end without Kaggle. Its numbers are stamped SYNTHETIC and
  must NOT be quoted in the README.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pacer.eval.baselines import STRATEGIES, PacingRunner, Strategy
from pacer.eval.budget_sizing import achievable_spend, size_budgets
from pacer.eval.delivery_metrics import delivery_report
from pacer.eval.pacing_metrics import pacing_report
from pacer.sim.burst import BurstConfig, inject_bursts
from pacer.sim.campaign import generate_campaigns
from pacer.sim.engine import Engine, EngineConfig
from pacer.sim.pid import PIDConfig
from pacer.sim.target import TrafficAwareTarget, UniformTarget
from pacer.sim.traffic import TrafficReplay, hourly_volume
from pacer.sim.tuning import grid_search

ARTIFACT_DIR = Path("artifacts")
ASSET_DIR = Path("../README_assets")
SPLIT_DIR = Path("data/splits")

DIURNAL = np.array(
    [2, 1, 1, 1, 2, 4, 7, 10, 12, 13, 13, 12, 11, 11, 12, 13, 14, 15, 16, 14, 11, 8, 5, 3],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_synthetic(n: int = 40000, seed: int = 0):
    rng = np.random.default_rng(seed)
    counts = np.maximum(1, (DIURNAL / DIURNAL.sum() * n).astype(int))
    hours = np.concatenate([[14100100 + h] * int(counts[h]) for h in range(24)])
    m = len(hours)
    bp = rng.choice(["0", "1"], size=m)
    sc = rng.choice(["a", "b", "c"], size=m)
    features = [{"banner_pos": bp[i], "site_category": sc[i]} for i in range(m)]
    pctrs = rng.beta(2, 8, m)
    labels = (rng.random(m) < pctrs).astype(np.int8)
    return hours, features, labels, pctrs


def build_real():
    import polars as pl

    from pacer.data.features import FeatureTransform
    from pacer.data.loader import _to_arrays
    from pacer.models.calibrate import IsotonicCalibrator, PlattCalibrator
    from pacer.models.export import onnx_predict

    test = pl.read_parquet(SPLIT_DIR / "test.parquet")
    transform = FeatureTransform.from_json(str(ARTIFACT_DIR / "feature_transform.json"))
    X, labels = _to_arrays(transform.transform(test), transform.field_order())
    pctrs = onnx_predict(str(ARTIFACT_DIR / "model.onnx"), X)

    cal_path = ARTIFACT_DIR / "calibrator.json"
    if cal_path.exists():
        d = json.loads(cal_path.read_text())
        if d["kind"] == "platt":
            cal = PlattCalibrator()
            cal.a, cal.b = d["a"], d["b"]
            pctrs = cal.predict(pctrs)
        else:
            cal = IsotonicCalibrator()
            import numpy as _np
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.X_thresholds_ = _np.array(d["x"])
            iso.y_thresholds_ = _np.array(d["y"])
            iso.X_min_, iso.X_max_ = d["x"][0], d["x"][-1]
            iso.increasing_ = True
            iso.f_ = None
            cal._iso = iso
            pctrs = cal.predict(pctrs)

    hours = test["hour"].to_numpy()
    features = test.select(["banner_pos", "site_category"]).to_dicts()
    return hours, features, np.asarray(labels), np.asarray(pctrs)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _feature_values(features):
    vals: dict[str, set] = {}
    for f in features:
        for k, v in f.items():
            vals.setdefault(k, set()).add(str(v))
    return {k: sorted(v) for k, v in vals.items()}


def run_one(
    strategy: Strategy,
    replay,
    features,
    labels,
    pctrs,
    counts,
    gains: PIDConfig,
    n_campaigns: int,
    bursty: bool,
    budgets: dict[int, float],
    seed: int,
):
    duration = replay.duration_seconds()
    curve = TrafficAwareTarget(counts) if strategy.traffic_aware else UniformTarget(duration)

    camps = generate_campaigns(n_campaigns, _feature_values(features), seed=seed)
    for c in camps:  # apply the inventory-sized budgets
        c.daily_budget = budgets[c.id]
    runner = PacingRunner(strategy, camps, curve, gains, seed=seed)
    eng = Engine(
        camps, features, labels, pctrs,
        EngineConfig(reserve=0.001, control_interval_s=gains.dt),
        control_hook=runner.control_hook,
        throttle_hook=runner.throttle_hook,
        bid_hook=runner.bid_hook,
    )
    stats = eng.run(replay)
    prep = pacing_report(stats.spend_trace, camps, curve)
    drep = delivery_report(stats)
    return {
        "strategy": strategy.name,
        "bursty": bursty,
        "mape_pct": prep.mean_abs_pacing_error_pct,
        "l2": prep.l2_spend_deviation,
        "util_mean": prep.utilization_mean,
        "early_exhaust": prep.early_exhaustion_frac,
        "clicks": drep.total_clicks,
        "clicks_per_dollar": drep.clicks_per_dollar,
        "ecpc": drep.effective_cpc,
    }, (stats, camps, curve)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--campaigns", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        print("=== SYNTHETIC DEMO DATA — numbers here are NOT for the README ===")
        hours, features, labels, pctrs = build_synthetic(seed=args.seed)
    else:
        if not (SPLIT_DIR / "test.parquet").exists():
            raise SystemExit(
                "No test split / model artifacts. Run `make data` and `make train` "
                "first, or use --synthetic for a demo run."
            )
        hours, features, labels, pctrs = build_real()

    _, counts = hourly_volume(hours)
    tune = grid_search(counts)
    gains = replace(tune.best, dt=10.0)
    print(f"tuned gains: kp={gains.kp} ki={gains.ki} kd={gains.kd} "
          f"(val pacing err {tune.best_error:.4f})")

    rows = []
    detail = {}
    for bursty in (False, True):
        replay = TrafficReplay(hours, seed=args.seed)
        if bursty:
            replay = inject_bursts(replay, BurstConfig(seed=args.seed))
        # calibrate budgets to inventory ONCE per traffic mode, reused by all
        # strategies so the comparison is at identical budgets.
        base = generate_campaigns(args.campaigns, _feature_values(features), seed=args.seed)
        ach = achievable_spend(base, features, labels, pctrs, replay)
        size_budgets(base, ach, target_utilization=0.6)
        budgets = {c.id: c.daily_budget for c in base}

        for strat in STRATEGIES:
            row, det = run_one(strat, replay, features, labels, pctrs, counts,
                               gains, args.campaigns, bursty, budgets, args.seed)
            rows.append(row)
            detail[(strat.name, bursty)] = det

    _print_table(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "sim_results.json").write_text(json.dumps(rows, indent=2))
    _make_plots(detail)


def _print_table(rows):
    print(f"\n{'strategy':<34}{'mode':<8}{'MAPE%':>8}{'util':>7}"
          f"{'earlyX':>8}{'clicks':>9}{'clk/$':>8}")
    for r in rows:
        mode = "bursty" if r["bursty"] else "smooth"
        print(f"{r['strategy']:<34}{mode:<8}{r['mape_pct']*100:>8.2f}"
              f"{r['util_mean']:>7.2f}{r['early_exhaust']:>8.2f}"
              f"{r['clicks']:>9}{r['clicks_per_dollar']:>8.3f}")


def _make_plots(detail):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing; skipping plots)")
        return

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. spend curve vs target for one representative campaign, smooth, overlaid
    fig, ax = plt.subplots(figsize=(9, 5))
    rep_id = None
    for (name, bursty), (stats, camps, curve) in detail.items():
        if bursty:
            continue
        if rep_id is None:
            # pick a mid-budget campaign present in every run
            rep_id = sorted(camps, key=lambda c: c.daily_budget)[len(camps) // 2].id
            tr = stats.spend_trace[rep_id]
            ts = [t for t, _ in tr]
            ax.plot(ts, [curve.spend_target(camps[rep_id].daily_budget, t) for t in ts],
                    "k--", label="target", lw=2)
        tr = stats.spend_trace[rep_id]
        ax.plot([t for t, _ in tr], [s for _, s in tr], label=name, lw=1)
    ax.set_xlabel("seconds"); ax.set_ylabel("cumulative spend")
    ax.set_title("Spend vs target — representative campaign (smooth)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ASSET_DIR / "spend_curves.png", dpi=110)

    # 2. utilization distribution per strategy (smooth)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, bursty), (stats, camps, curve) in detail.items():
        if bursty:
            continue
        util = [c.spend / c.daily_budget for c in camps]
        ax.hist(util, bins=30, histtype="step", label=name)
    ax.set_xlabel("budget utilization"); ax.set_ylabel("campaigns")
    ax.set_title("Budget utilization across campaigns (smooth)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ASSET_DIR / "utilization_dist.png", dpi=110)

    # 3. anti-windup multiplier response
    from pacer.sim.windup_demo import run_windup_scenario

    on = run_windup_scenario(anti_windup=True)
    off = run_windup_scenario(anti_windup=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(on.mults, label="anti-windup ON")
    ax.plot(off.mults, label="anti-windup OFF")
    ax.set_xlabel("control tick"); ax.set_ylabel("pacing multiplier")
    ax.set_title("Anti-windup: multiplier response to a traffic trough")
    ax.legend()
    fig.tight_layout(); fig.savefig(ASSET_DIR / "anti_windup.png", dpi=110)
    print(f"wrote plots -> {ASSET_DIR}/")


if __name__ == "__main__":
    main()

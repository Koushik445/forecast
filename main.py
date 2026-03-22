"""
main.py
=======
Single end-to-end entry point.

Usage
-----
    # Run full pipeline (load → preprocess → train → predict)
    python main.py

    # Skip training, only re-generate predictions (must have trained models saved)
    python main.py --predict-only

    # Custom data directory
    CFL_DATA_DIR=/path/to/data python main.py

    # Full help
    python main.py --help

Output
------
    outputs/forecast_FY26Q2.csv         ← Final forecast
    outputs/cv_summary.csv              ← Cross-validation performance
    outputs/feature_importance.csv      ← Top features from best model
    outputs/panel_clean.csv             ← (optional) cleaned merged panel
    models/                             ← Saved model files
"""

import os
import sys
import argparse
import logging
import time

import numpy as np
import pandas as pd

# ── Make sure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils             import setup_logging, print_section, print_forecast_table, print_cv_summary
from data_loading      import load_all
from preprocessing     import preprocess
from feature_engineering import engineer_features, get_feature_cols
import train  as tr
import predict as pr

OUTPUTS_DIR = os.environ.get("CFL_OUTPUTS_DIR", "outputs")
MODELS_DIR  = os.environ.get("CFL_MODELS_DIR",  "models")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CFL Demand Forecasting Pipeline")
    p.add_argument("--predict-only", action="store_true",
                   help="Skip training; load saved models and generate predictions.")
    p.add_argument("--no-save-panel", action="store_true",
                   help="Do not save the intermediate panel CSV.")
    p.add_argument("--cv-splits", type=int, default=4,
                   help="Number of walk-forward CV folds (default: 4)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity")
    return p.parse_args()


# ── Step banners ──────────────────────────────────────────────────────────────

def _step(n: int, title: str):
    log = logging.getLogger(__name__)
    log.info(f"\n{'='*60}\n  STEP {n}: {title}\n{'='*60}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    log = logging.getLogger(__name__)
    t0  = time.time()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 — Data Loading
    # ────────────────────────────────────────────────────────────────────────
    _step(1, "DATA LOADING")
    data = load_all()
    print_section("Dataset Summary")
    for name, df in data.items():
        log.info(f"  {name:20s}: {df.shape}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 — Preprocessing
    # ────────────────────────────────────────────────────────────────────────
    _step(2, "PREPROCESSING & MERGING")
    panel = preprocess(data)
    log.info(f"Panel shape after preprocessing: {panel.shape}")

    if not args.no_save_panel:
        panel_path = os.path.join(OUTPUTS_DIR, "panel_clean.csv")
        panel.to_csv(panel_path, index=False)
        log.info(f"  Panel saved to {panel_path}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3 — Feature Engineering
    # ────────────────────────────────────────────────────────────────────────
    _step(3, "FEATURE ENGINEERING")
    feats = engineer_features(panel)
    feature_cols = get_feature_cols(feats)
    log.info(f"  {len(feature_cols)} features across {feats['product_name'].nunique()} products")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4 — Training (or load)
    # ────────────────────────────────────────────────────────────────────────
    if args.predict_only:
        _step(4, "LOADING TRAINED MODELS (--predict-only)")
        trained = tr.load_trained_models()
    else:
        _step(4, "MODEL TRAINING & WALK-FORWARD CV")
        trained = tr.train_all_models(feats, feature_cols, n_cv_splits=args.cv_splits)

    # Print CV summary
    print_cv_summary(trained["cv_scores"])

    # Save CV summary
    cv_rows = []
    for name, scores in trained["cv_scores"].items():
        cv_rows.append({
            "model":       name,
            "wmape_mean":  scores["wmape_mean"],
            "wmape_std":   scores["wmape_std"],
            "accuracy":    1 - scores["wmape_mean"],
            "bias_mean":   scores["bias_mean"],
            "weight":      trained["weights"].get(name, 0),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(OUTPUTS_DIR, "cv_summary.csv"), index=False)

    # Save feature importances
    print_section("Feature Importances (Best Model)")
    best_name  = min(trained["cv_scores"],
                     key=lambda n: trained["cv_scores"][n]["wmape_mean"])
    best_model = trained["models"][best_name]
    fi = None
    try:
        importances = best_model.feature_importances_
        fi = pd.DataFrame({
            "feature":    trained["feature_cols"],
            "importance": importances
        }).sort_values("importance", ascending=False)
        log.info(f"\n  Top 20 features from '{best_name}':")
        for _, row in fi.head(20).iterrows():
            log.info(f"    {row['feature']:40s}: {row['importance']:.4f}")
        fi.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False)
    except AttributeError:
        log.warning("  Feature importances not available for this model type.")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5 — Prediction
    # ────────────────────────────────────────────────────────────────────────
    _step(5, "GENERATING FORECAST (FY26 Q2)")
    forecast = pr.predict(feats, trained)
    print_forecast_table(forecast)

    # ── Also produce a comparison vs competitor forecasts ──
    comp = data["competitor_fcsts"].copy()
    comp["product_name"] = comp["product_name"].str.strip()
    summary = forecast.merge(
        comp[["product_name", "demand_planners_fcst", "marketing_fcst", "datascience_fcst"]],
        on="product_name", how="left"
    )
    summary["our_vs_dp"]  = (
        (summary["forecast_units"] - summary["demand_planners_fcst"])
        / (summary["demand_planners_fcst"] + 1e-6) * 100
    ).round(1)

    print_section("Our Forecast vs Competitor Benchmarks")
    print(summary[["product_name", "forecast_units",
                    "demand_planners_fcst", "marketing_fcst", "datascience_fcst",
                    "our_vs_dp"]].to_string(index=False))

    summary.to_csv(os.path.join(OUTPUTS_DIR, "forecast_comparison.csv"), index=False)

    elapsed = time.time() - t0
    print_section(f"Pipeline Complete in {elapsed:.1f}s")
    log.info(f"  Forecast -> {OUTPUTS_DIR}/forecast_FY26Q2.csv")
    log.info(f"  Compare  -> {OUTPUTS_DIR}/forecast_comparison.csv")
    log.info(f"  CV       -> {OUTPUTS_DIR}/cv_summary.csv")
    log.info(f"  Features -> {OUTPUTS_DIR}/feature_importance.csv")

    return forecast


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    setup_logging(
        level=args.log_level,
        log_file=os.path.join("logs", "pipeline.log")
    )
    run_pipeline(args)
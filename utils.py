"""
utils.py
========
Shared utilities for logging, plotting, reporting, and evaluation.
"""

import os
import logging
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = None):
    import sys
    # Force UTF-8 on Windows to avoid cp1252 encoding errors with special chars
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    handlers = []
    # Stream handler with explicit UTF-8
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    handlers.append(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True,
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def wmape(actual, predicted):
    actual    = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    return np.sum(np.abs(actual - predicted)) / denom if denom > 0 else np.nan


def bias(actual, predicted):
    actual    = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    return (np.sum(predicted) - np.sum(actual)) / denom if denom > 0 else 0.0


def mae(actual, predicted):
    return np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))


def eval_report(actual, predicted, name: str = "Model") -> dict:
    wm = wmape(actual, predicted)
    bi = bias(actual, predicted)
    return {
        "model":    name,
        "WMAPE":    round(wm,  4),
        "Accuracy": round(1 - wm, 4),
        "Bias":     round(bi, 4),
        "MAE":      round(mae(actual, predicted), 1),
    }


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_cols: list, top_n: int = 30) -> pd.DataFrame:
    """Extract feature importances from sklearn-style or lgbm models."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df.head(top_n)


# ── Print / display helpers ───────────────────────────────────────────────────

def print_section(title: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_forecast_table(forecast_df: pd.DataFrame):
    """Pretty-print the forecast output."""
    print_section("FORECAST — FY26 Q2 (Units)")
    disp = forecast_df[["product_name", "forecast_units", "lifecycle",
                         "last2q_avg", "vs_last2q_avg_pct"]].copy()
    disp["last2q_avg"]       = disp["last2q_avg"].round(0).astype("Int64")
    disp["vs_last2q_avg_pct"] = disp["vs_last2q_avg_pct"].map(lambda x: f"{x:+.1f}%")
    print(disp.to_string(index=False))


def print_cv_summary(cv_scores: dict):
    print_section("Walk-Forward CV Summary")
    rows = []
    for name, scores in cv_scores.items():
        rows.append({
            "Model":      name,
            "WMAPE Mean": f"{scores['wmape_mean']:.4f}",
            "WMAPE Std":  f"{scores['wmape_std']:.4f}",
            "Bias Mean":  f"{scores['bias_mean']:.4f}",
            "Accuracy":   f"{(1 - scores['wmape_mean']):.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    logging.getLogger(__name__).info(f"Saved: {path}")


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ── Reproducibility seed ──────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ── Data sanity checks ────────────────────────────────────────────────────────

def sanity_check_panel(panel: pd.DataFrame):
    log = logging.getLogger(__name__)
    n_products = panel["product_name"].nunique()
    n_quarters = panel["quarter"].nunique()
    n_missing  = panel["actual_units"].isna().sum()
    log.info(f"Panel: {n_products} products × {n_quarters} quarters = {len(panel)} rows")
    log.info(f"  Missing actual_units: {n_missing} ({n_missing / len(panel) * 100:.1f}%)")
    if n_missing > len(panel) * 0.3:
        log.warning("  ⚠️  More than 30% of actual_units are missing — check data loading!")


if __name__ == "__main__":
    print("Utils module loaded successfully.")
    print(eval_report([100, 200, 300], [110, 190, 310], "TestModel"))
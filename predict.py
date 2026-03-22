"""
predict.py
==========
Generate forecasts for the target quarter (FY26 Q2 = CY 2025Q4).

Steps
-----
1. Load trained ensemble from disk
2. Build the "future row" for each product:
     - Use known lag features from the most recent actual quarter
     - Forward-fill external signals (VMS / SCMS / Big Deal) from last known quarter
3. Generate predictions from each model
4. Weighted ensemble average
5. Apply bias correction (additive, in log space → multiplicative in units)
6. Output clean forecast DataFrame and CSV
"""

import os
import logging
import numpy as np
import pandas as pd

import train as tr
import feature_engineering as fe
from preprocessing import normalise_period, quarter_to_date

log = logging.getLogger(__name__)

OUTPUTS_DIR = os.environ.get("CFL_OUTPUTS_DIR", "outputs")
FUTURE_QTR  = fe.FUTURE_QTR    # CY 2025Q4


# ── Build future rows ─────────────────────────────────────────────────────────

def _most_recent_row(df: pd.DataFrame, product: str) -> pd.Series:
    """Return the last available row for a product."""
    prod_df = df[df["product_name"] == product].sort_values("date")
    return prod_df.iloc[-1]


def build_future_df(feats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one synthetic row per product for the target quarter.

    All lag/rolling features are shifted forward by exactly 1 period
    so the model sees the same feature structure at prediction time
    as it did during training.
    """
    future_date = quarter_to_date(FUTURE_QTR)
    products    = feats_df["product_name"].unique()
    rows        = []

    cq_to_fq = {1: 3, 2: 4, 3: 1, 4: 2}

    for prod in products:
        prod_df = feats_df[feats_df["product_name"] == prod].sort_values("date")
        if len(prod_df) == 0:
            continue

        last     = prod_df.iloc[-1].copy()
        actuals  = prod_df["actual_units"].values   # historical series

        # ── 1. Time features ──────────────────────────────────────────────────
        last["quarter"]       = FUTURE_QTR
        last["date"]          = future_date
        last["cal_year"]      = future_date.year
        last["cal_month"]     = future_date.month
        last["cal_quarter"]   = future_date.quarter
        last["fiscal_quarter"]= cq_to_fq[future_date.quarter]
        last["time_idx"]      = prod_df["time_idx"].max() + 1
        last["quarter_sin"]   = np.sin(2 * np.pi * future_date.quarter / 4)
        last["quarter_cos"]   = np.cos(2 * np.pi * future_date.quarter / 4)

        # ── 2. Fiscal quarter dummies ─────────────────────────────────────────
        fq_val = cq_to_fq[future_date.quarter]
        for q in [1, 2, 3, 4]:
            col = f"fq_{q}"
            if col in last.index:
                last[col] = int(q == fq_val)

        # ── 3. Lag features (shift forward by 1) ─────────────────────────────
        # New lag_1 = last known actual (index -1)
        # New lag_2 = previous lag_1   (index -2), etc.
        last_actual = actuals[-1] if len(actuals) > 0 else 0.0
        for lag in [8, 6, 4, 3, 2, 1]:
            lag_col = f"lag_{lag}_actual_units"
            if lag_col in last.index:
                # lag_1 for future = last actual; lag_2 = previous lag_1, etc.
                idx = -lag   # index into actuals array
                if abs(idx) <= len(actuals):
                    last[lag_col] = actuals[idx] if abs(idx) < len(actuals) else np.nan
                else:
                    last[lag_col] = np.nan

        # ── 4. Rolling features (recompute from shifted series) ───────────────
        # Append 0 as placeholder then compute rolling — mimics training behaviour
        extended = np.append(actuals, np.nan)   # nan for unknown future value
        for w in [3, 4, 6]:
            mean_col = f"roll_mean_{w}_actual_units"
            std_col  = f"roll_std_{w}_actual_units"
            window   = extended[-(w+1):-1]       # last w actuals (no future)
            window   = window[~np.isnan(window)]
            if mean_col in last.index:
                last[mean_col] = np.mean(window) if len(window) > 0 else 0.0
            if std_col in last.index:
                last[std_col]  = np.std(window)  if len(window) > 1 else 0.0

        # ── 5. Trend features ─────────────────────────────────────────────────
        if len(actuals) >= 2:
            qoq = actuals[-1] - actuals[-2]
            prev_qoq = (actuals[-2] - actuals[-3]) if len(actuals) >= 3 else 0.0
            last["qoq_change"]     = qoq
            last["qoq_pct"]        = qoq / (actuals[-2] + 1e-6)
            last["qoq_acceleration"] = qoq - prev_qoq
            last["qoq_pct_prev"]   = prev_qoq / (actuals[-3] + 1e-6) if len(actuals) >= 3 else 0.0
            last["growth_accel"]   = last["qoq_pct"] - last["qoq_pct_prev"]

        if len(actuals) >= 5:
            last["yoy_change"] = actuals[-1] - actuals[-5]
            last["yoy_ratio"]  = (actuals[-1] + 1) / (actuals[-5] + 1)

        # Consecutive streak
        if len(actuals) >= 2:
            streak = 0
            for i in range(1, len(actuals)):
                d = actuals[i] - actuals[i-1]
                if d > 0:
                    streak = max(streak, 0) + 1
                elif d < 0:
                    streak = min(streak, 0) - 1
                else:
                    streak = 0
            last["demand_streak"] = streak

        # Log-space trend slope
        log_actuals = np.log1p(actuals.clip(0))
        if len(log_actuals) >= 4:
            y = log_actuals[-4:].astype(float)
            x = np.arange(4, dtype=float) - 1.5
            denom = (x**2).sum()
            last["log_trend_slope"] = float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0

        # Recompute trend slope over last 4 actuals
        if len(actuals) >= 4:
            y = actuals[-4:].astype(float)
            x = np.arange(4, dtype=float) - 1.5
            denom = (x**2).sum()
            last["trend_slope_4q"] = float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0

        # Peak features
        last["cum_max_units"]      = float(np.nanmax(actuals))
        last["pct_of_peak"]        = last_actual / (last["cum_max_units"] + 1e-6)
        last["dist_from_peak"]     = last["cum_max_units"] - last_actual
        last["dist_from_peak_pct"] = last["dist_from_peak"] / (last["cum_max_units"] + 1e-6)

        # Volatility
        if len(actuals) >= 4:
            w4 = actuals[-4:]
            m4 = w4.mean()
            last["cv_4q"] = w4.std() / (m4 + 1e-6) if m4 > 0 else 0.0

        # hist_fq_mean: mean of same fiscal quarter historically
        # fiscal_quarter was dummified - recover it from fq_ columns
        fq_col = f"fq_{fq_val}"
        if fq_col in prod_df.columns:
            fq_mask = prod_df[fq_col] == 1
        elif "fiscal_quarter" in prod_df.columns:
            fq_mask = prod_df["fiscal_quarter"] == fq_val
        else:
            fq_mask = pd.Series([False] * len(prod_df), index=prod_df.index)
        fq_vals = prod_df.loc[fq_mask, "actual_units"].dropna()
        last["hist_fq_mean"]    = float(fq_vals.mean()) if len(fq_vals) > 0 else last_actual
        last["units_vs_hist_fq"]= last_actual / (last["hist_fq_mean"] + 1e-6)

        # ── 6. External lags (use the most recent known values) ───────────────
        for ext_col in ["vms_total", "scms_total", "scms_enterprise",
                        "big_deal_units", "big_deal_share"]:
            lag1_col = f"lag1_{ext_col}"
            if lag1_col in last.index and ext_col in prod_df.columns:
                last[lag1_col] = prod_df[ext_col].iloc[-1]

        # ── 7. Portfolio share — use last known product share ─────────────────
        # product_share is already set from last row; it's lag-1 based so safe

        # ── 9. Target encoding features ───────────────────────────────────────
        log_actuals_arr = np.log1p(np.maximum(actuals, 0))
        te_mean = float(log_actuals_arr.mean()) if len(log_actuals_arr) > 0 else 0.0
        te_std  = float(log_actuals_arr.std())  if len(log_actuals_arr) > 1 else 0.0

        if "te_product_mean" in last.index:
            last["te_product_mean"]   = te_mean
        if "te_product_std" in last.index:
            last["te_product_std"]    = te_std
        if "te_product_global" in last.index:
            last["te_product_global"] = te_mean

        # Seasonal te: use value already on the last row (correct fq mean)
        # fq_col = f"te_prod_fq{fq_val}" already populated from last row copy

        # ── 10. Jensen correction ──────────────────────────────────────────────
        raw_correction = float(np.var(log_actuals_arr) * 0.5) if len(log_actuals_arr) > 1 else 0.0
        last["log_var_correction"] = min(raw_correction, 0.25)   # same cap as training

        # ── 11. Target: unknown ───────────────────────────────────────────────
        last[fe.TARGET_RAW] = np.nan
        last[fe.TARGET_LOG] = np.nan

        rows.append(last)

    future_df = pd.DataFrame(rows).reset_index(drop=True)
    log.info(f"Built {len(future_df)} future rows for quarter {FUTURE_QTR}")
    return future_df


# ── Ensemble predict ───────────────────────────────────────────────────────────

def ensemble_predict(future_df: pd.DataFrame,
                     models: dict,
                     weights: dict,
                     feature_cols: list,
                     bias_corrections: dict = None) -> np.ndarray:
    """
    Weighted ensemble prediction in raw units.

    Steps per model:
      1. Predict log1p(units)
      2. Apply Jensen's inequality correction (+0.5 * log_var per product)
         to correct the systematic under-prediction from log-transform
      3. Apply CV-measured bias correction in log space
      4. Convert to raw units via expm1
      5. Weighted average across models
    """
    X = future_df[feature_cols].fillna(0).values
    bias_corrections = bias_corrections or {}

    # Jensen correction per product — stored in feature matrix if available
    jensen_col = "log_var_correction"
    if jensen_col in future_df.columns:
        jensen_correction = future_df[jensen_col].fillna(0).values
    else:
        jensen_correction = np.zeros(len(future_df))

    weighted_sum = np.zeros(len(future_df))
    total_weight = 0.0

    for name, model in models.items():
        pred_log = model.predict(X)

        # 1. Jensen correction (per-product, addresses log-transform bias)
        pred_log = pred_log + jensen_correction

        # 2. CV-measured bias correction in log space
        bias = bias_corrections.get(name, 0.0)
        if abs(bias) > 0.005:
            denom = 1.0 + bias
            if denom > 0.05:
                pred_log = pred_log + np.log(1.0 / denom)

        pred_raw = np.expm1(np.clip(pred_log, 0, 20))

        w = weights.get(name, 1.0 / len(models))
        weighted_sum += w * pred_raw
        total_weight += w
        log.debug(f"  {name}: mean_pred={pred_raw.mean():.0f}, weight={w:.3f}")

    ensemble_preds = weighted_sum / (total_weight + 1e-9)
    ensemble_preds = np.clip(ensemble_preds, 0, None)
    return ensemble_preds


# ── Post-processing: round and sanity-check ───────────────────────────────────

def postprocess_predictions(preds: np.ndarray,
                             products: list,
                             feats_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Round to nearest integer
    - Flag NPI products with very small predictions (expected)
    - Return a clean submission DataFrame
    """
    result = pd.DataFrame({
        "product_name":   products,
        "forecast_units": np.round(preds).astype(int),
    })

    # Attach lifecycle for context — use whatever metadata columns survived
    meta_cols = ["product_name", "lifecycle"]
    if "product_category" in feats_df.columns:
        meta_cols.append("product_category")
    meta = feats_df[meta_cols].drop_duplicates()
    result = result.merge(meta, on="product_name", how="left")

    # Sanity: floor predictions at 0
    result["forecast_units"] = result["forecast_units"].clip(lower=0)

    # Compare to trailing 2-quarter average
    last2 = (
        feats_df[feats_df["quarter"] != FUTURE_QTR]
        .sort_values(["product_name", "date"])
        .groupby("product_name")
        .apply(lambda g: g["actual_units"].iloc[-2:].mean()
               if len(g) >= 2 else g["actual_units"].iloc[-1:].mean())
        .reset_index()
        .rename(columns={0: "last2q_avg"})
    )
    result = result.merge(last2, on="product_name", how="left")
    result["vs_last2q_avg_pct"] = (
        (result["forecast_units"] - result["last2q_avg"])
        / (result["last2q_avg"] + 1e-6) * 100
    ).round(1)

    return result


# ── Main predict function ────────────────────────────────────────────────────

def predict(feats_df: pd.DataFrame,
            trained: dict = None) -> pd.DataFrame:
    """
    Full prediction pipeline with hybrid ML + statistical blend.

    Final prediction =
        w_ml   × ensemble_ML_prediction
      + w_roll4 × rolling_4q_mean
      + w_lag1  × lag_1_actual
    """
    if trained is None:
        log.info("Loading trained models from disk ...")
        trained = tr.load_trained_models()

    models           = trained["models"]
    weights          = trained["weights"]
    feature_cols     = trained["feature_cols"]
    bias_corrections = trained.get("bias_corrections", {})
    hybrid_weights   = trained.get("hybrid_weights", {"ml": 0.65, "roll4": 0.25, "lag1": 0.10})
    product_bias     = trained.get("product_bias", {})

    # Build future rows
    future_df = build_future_df(feats_df)

    # ── ML ensemble prediction ──────────────────────────────────────────────
    log.info("Generating ensemble predictions ...")
    ml_preds = ensemble_predict(future_df, models, weights, feature_cols, bias_corrections)

    # ── Statistical baselines ────────────────────────────────────────────────
    roll4 = future_df["roll_mean_4_actual_units"].fillna(
                future_df.get("lag_1_actual_units", pd.Series(0, index=future_df.index))
            ).fillna(0).values
    lag1  = future_df["lag_1_actual_units"].fillna(0).values if "lag_1_actual_units" in future_df.columns \
            else np.zeros(len(future_df))

    # ── Hybrid blend ─────────────────────────────────────────────────────────
    w_ml   = hybrid_weights["ml"]
    w_roll = hybrid_weights["roll4"]
    w_lag  = hybrid_weights["lag1"]

    final_preds = w_ml * ml_preds + w_roll * roll4 + w_lag * lag1
    final_preds = np.maximum(final_preds, 0)

    # ── Per-product bias correction ──────────────────────────────────────────
    products = future_df["product_name"].tolist()
    n_corrected = 0
    for i, prod in enumerate(products):
        bi = product_bias.get(prod, 0.0)
        if abs(bi) > 0.05:   # only correct when bias is meaningful
            mult = 1.0 / (1.0 + bi)
            corrected = final_preds[i] * mult
            # Guardrail: prediction cannot exceed 2.5x the most recent actual
            # This prevents extreme overcorrection on volatile/NPI products
            lag1_val = float(future_df["lag_1_actual_units"].iloc[i]) \
                       if "lag_1_actual_units" in future_df.columns else 0.0
            if lag1_val > 1.0:
                corrected = min(corrected, 2.5 * lag1_val)
            final_preds[i] = corrected
            n_corrected += 1
            log.debug(f"  {prod}: bias={bi:+.3f} -> x{mult:.3f} -> {final_preds[i]:.0f}")
    log.info(f"Per-product bias correction applied to {n_corrected}/{len(products)} products")

    log.info(f"Hybrid blend: ML={w_ml:.2f} x {ml_preds.mean():.0f} + "
             f"Roll4={w_roll:.2f} x {roll4.mean():.0f} + "
             f"Lag1={w_lag:.2f} x {lag1.mean():.0f} = {final_preds.mean():.0f} avg")

    # Post-process
    result = postprocess_predictions(final_preds, future_df["product_name"].tolist(), feats_df)

    # Save output
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "forecast_FY26Q2.csv")
    result.to_csv(out_path, index=False)
    log.info(f"Forecast saved to {out_path}")

    return result


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loading        import load_all
    from preprocessing       import preprocess
    from feature_engineering import engineer_features, get_feature_cols

    data    = load_all()
    panel   = preprocess(data)
    feats   = engineer_features(panel)

    result  = predict(feats)
    print("\n=== Forecast FY26 Q2 ===")
    print(result[["product_name", "forecast_units", "lifecycle",
                  "last2q_avg", "vs_last2q_avg_pct"]].to_string(index=False))
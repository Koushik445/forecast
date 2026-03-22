"""
feature_engineering.py
======================
Builds a rich feature matrix from the clean panel DataFrame.

Features generated:
  ▸ Temporal:    quarter number, month, fiscal quarter, year, trend index
  ▸ Lag:         lag-1, 2, 3, 4 (quarterly) actual_units per product
  ▸ Rolling:     rolling mean/std 3q, 4q per product
  ▸ YoY:         year-over-year change (lag-4), YoY ratio
  ▸ Seasonality: quarter-of-year dummies, fiscal quarter code
  ▸ Growth:      % change vs prior quarter, vs 2q ago
  ▸ External:    VMS features, SCMS features, Big Deal features
  ▸ Product:     lifecycle code, category code, product type dummies
  ▸ Aggregate:   portfolio total, product share of portfolio
  ▸ Competitor:  demand planners / marketing / DS team forecasts (when available)
  ▸ Log target:  log1p(actual_units) — used for training

All lag/rolling features are computed strictly within each product group
to prevent cross-product leakage.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

TARGET_RAW  = "actual_units"
TARGET_LOG  = "log_units"
FUTURE_QTR  = "2025Q4"   # CY equivalent of FY26 Q2


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_pct(a, b):
    """Percentage change from b to a, with zero guard."""
    return np.where(b == 0, 0.0, (a - b) / (b + 1e-9))


# ── temporal features ─────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cal_year"]    = df["date"].dt.year
    df["cal_month"]   = df["date"].dt.month
    df["cal_quarter"] = df["date"].dt.quarter     # 1-4

    # Cisco fiscal quarter (FY starts Aug)
    # cal Q3 → FQ1, Q4 → FQ2, Q1 → FQ3, Q2 → FQ4
    cq_to_fq = {1: 3, 2: 4, 3: 1, 4: 2}
    df["fiscal_quarter"] = df["cal_quarter"].map(cq_to_fq)

    # Monotonic time index (useful as trend feature)
    dates_sorted = sorted(df["date"].unique())
    date_rank    = {d: i for i, d in enumerate(dates_sorted)}
    df["time_idx"] = df["date"].map(date_rank)

    # Sine/cosine encoding of quarter for seasonality (circular)
    df["quarter_sin"] = np.sin(2 * np.pi * df["cal_quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["cal_quarter"] / 4)

    return df


# ── lag features ──────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     lags: list = [1, 2, 3, 4, 6, 8],
                     col: str = TARGET_RAW) -> pd.DataFrame:
    """
    Add lag features within each product group.
    df must be sorted by (product_name, date).
    """
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)

    for lag in lags:
        lag_col = f"lag_{lag}_{col}"
        df[lag_col] = (
            df.groupby("product_name")[col]
            .shift(lag)
        )

    log.debug(f"  Added lags {lags} for '{col}'")
    return df


# ── rolling statistics ────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                         windows: list = [3, 4, 6],
                         col: str = TARGET_RAW) -> pd.DataFrame:
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)

    for w in windows:
        # Rolling mean and std — shift(1) to avoid look-ahead
        rolled_mean = (
            df.groupby("product_name")[col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean())
        )
        rolled_std = (
            df.groupby("product_name")[col]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=max(1, w // 2)).std())
        )
        df[f"roll_mean_{w}_{col}"] = rolled_mean
        df[f"roll_std_{w}_{col}"]  = rolled_std.fillna(0)

    log.debug(f"  Added rolling windows {windows} for '{col}'")
    return df


# ── trend & momentum ──────────────────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame,
                       col: str = TARGET_RAW) -> pd.DataFrame:
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)

    prev1 = df.groupby("product_name")[col].shift(1)
    prev2 = df.groupby("product_name")[col].shift(2)
    prev3 = df.groupby("product_name")[col].shift(3)
    prev4 = df.groupby("product_name")[col].shift(4)

    # ── Basic momentum ──────────────────────────────────────────────────────
    df["qoq_change"] = df[col] - prev1
    df["qoq_pct"]    = _safe_pct(df[col].values, prev1.values)
    df["yoy_change"] = df[col] - prev4
    df["yoy_ratio"]  = (df[col] + 1) / (prev4 + 1)

    # ── Acceleration: is the trend speeding up or slowing? ─────────────────
    # (this quarter's change) - (last quarter's change) — key for NPI ramps
    prev_qoq = prev1 - prev2
    df["qoq_acceleration"] = df["qoq_change"] - prev_qoq

    # Growth rate of the growth rate — captures exponential NPI ramps
    df["qoq_pct_prev"] = _safe_pct(prev1.values, prev2.values)
    df["growth_accel"] = df["qoq_pct"] - df["qoq_pct_prev"]

    # ── Consecutive decline / growth streak ─────────────────────────────────
    # Positive = n consecutive quarters of growth; negative = decline streak
    def _streak(s: pd.Series) -> pd.Series:
        s_shift = s.shift(1)
        direction = np.sign(s - s_shift)    # +1, 0, -1 per quarter
        result = []
        streak = 0
        for d in direction:
            if pd.isna(d) or d == 0:
                streak = 0
            elif d > 0:
                streak = max(streak, 0) + 1
            else:
                streak = min(streak, 0) - 1
            result.append(streak)
        return pd.Series(result, index=s.index)

    df["demand_streak"] = df.groupby("product_name")[col].transform(_streak)

    # ── Log-space trend slope (better for multiplicative trends) ───────────
    log_col = np.log1p(df[col].clip(lower=0))
    df["_log_col_tmp"] = log_col

    def _log_slope(s: pd.Series, window: int = 4) -> pd.Series:
        def _calc(arr):
            n = len(arr)
            if n < 2:
                return 0.0
            x = np.arange(n, dtype=float) - (n - 1) / 2
            y = np.array(arr, dtype=float)
            y -= y.mean()
            denom = (x**2).sum()
            return (x * y).sum() / denom if denom > 0 else 0.0
        return s.shift(1).rolling(window, min_periods=2).apply(_calc, raw=True)

    df["log_trend_slope"] = df.groupby("product_name")["_log_col_tmp"].transform(_log_slope)
    df = df.drop(columns=["_log_col_tmp"])

    # ── Linear trend slope in raw space ────────────────────────────────────
    def _slope(s: pd.Series, window: int = 4) -> pd.Series:
        def _calc(arr):
            n = len(arr)
            if n < 2:
                return 0.0
            x = np.arange(n, dtype=float)
            x -= x.mean()
            y = np.array(arr, dtype=float)
            y -= y.mean()
            denom = (x ** 2).sum()
            return (x * y).sum() / denom if denom > 0 else 0.0
        return s.shift(1).rolling(window, min_periods=2).apply(_calc, raw=True)

    df["trend_slope_4q"] = df.groupby("product_name")[col].transform(_slope)

    # ── Peak / trough features ──────────────────────────────────────────────
    df["cum_max_units"] = df.groupby("product_name")[col].transform(
        lambda s: s.shift(1).expanding().max()
    )
    df["pct_of_peak"] = df[col] / (df["cum_max_units"] + 1e-6)

    # Distance from peak — captures decline magnitude signal
    df["dist_from_peak"] = df["cum_max_units"] - df[col]
    df["dist_from_peak_pct"] = df["dist_from_peak"] / (df["cum_max_units"] + 1e-6)

    # ── Volatility: CV of last 4 quarters ──────────────────────────────────
    roll4_mean = df.groupby("product_name")[col].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean()
    )
    roll4_std = df.groupby("product_name")[col].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    df["cv_4q"] = (roll4_std / (roll4_mean + 1e-6)).fillna(0)

    return df


# ── YoY / Seasonal normalisation ─────────────────────────────────────────────

def add_seasonal_features(df: pd.DataFrame,
                           col: str = TARGET_RAW) -> pd.DataFrame:
    """
    For each product×quarter, compute the historical mean for that fiscal quarter
    (e.g., "what are FQ1 sales historically?") — captures seasonality.
    """
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)

    # Mean of same fiscal quarter in prior years (expanding, no look-ahead)
    def _hist_fq_mean(group):
        out = []
        seen = {}
        for _, row in group.iterrows():
            fq = row["fiscal_quarter"]
            prev_vals = seen.get(fq, [])
            out.append(np.mean(prev_vals) if prev_vals else np.nan)
            seen.setdefault(fq, []).append(row[col])
        return pd.Series(out, index=group.index)

    df["hist_fq_mean"] = df.groupby("product_name", group_keys=False).apply(_hist_fq_mean)
    df["hist_fq_mean"] = df.groupby("product_name")["hist_fq_mean"].transform(
        lambda s: s.ffill()
    )
    df["units_vs_hist_fq"] = df[col] / (df["hist_fq_mean"] + 1e-6)

    return df


# ── Portfolio / cross-product features ───────────────────────────────────────

def add_portfolio_features(df: pd.DataFrame,
                            col: str = TARGET_RAW) -> pd.DataFrame:
    """
    Portfolio-level aggregations per quarter.
    IMPORTANT: All shares computed from LAG-1 values to avoid leakage.
    """
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)

    # Lag-1 actual per product (no current-quarter leakage)
    df["_lag1_for_share"] = df.groupby("product_name")[col].shift(1)

    # Portfolio total from lag-1 values
    port_lag = (
        df.groupby("quarter")["_lag1_for_share"]
        .sum()
        .reset_index()
        .rename(columns={"_lag1_for_share": "portfolio_total"})
    )

    # Category total from lag-1 values  
    cat_lag = (
        df.groupby(["quarter", "product_category"])["_lag1_for_share"]
        .sum()
        .reset_index()
        .rename(columns={"_lag1_for_share": "category_total"})
    )

    df = df.merge(port_lag, on="quarter", how="left")
    df = df.merge(cat_lag,  on=["quarter", "product_category"], how="left")

    # Shares based on lag-1 — safe for both training and prediction
    df["product_share"]  = df["_lag1_for_share"] / (df["portfolio_total"] + 1e-6)
    df["category_share"] = df["_lag1_for_share"] / (df["category_total"] + 1e-6)

    df = df.drop(columns=["_lag1_for_share"])
    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target encoding: replace product_id with statistically meaningful signals.

    Features added:
      te_product_mean  — expanding mean of log_units per product (no look-ahead)
      te_product_std   — expanding std  of log_units per product
      te_prod_fq{n}    — expanding mean of log_units for each fiscal quarter × product
                         (captures per-product seasonality pattern directly)

    All computed with shift(1) to prevent leakage.
    """
    df = df.copy()

    df["te_product_mean"] = (
        df.groupby("product_name")[TARGET_LOG]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["te_product_std"] = (
        df.groupby("product_name")[TARGET_LOG]
        .transform(lambda s: s.shift(1).expanding().std().fillna(0))
    )

    # Seasonal target encoding: product × fiscal_quarter
    # Uses the dummy columns that already exist after add_categorical_dummies()
    # OR computes on fiscal_quarter before dummies are applied
    if "fiscal_quarter" in df.columns:
        for fq in [1, 2, 3, 4]:
            enc_col = f"te_prod_fq{fq}"
            mask    = df["fiscal_quarter"] == fq
            df[enc_col] = np.nan
            df.loc[mask, enc_col] = (
                df[mask].groupby("product_name")[TARGET_LOG]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
            # Forward-fill within product for quarters where this FQ hasn't occurred yet
            df[enc_col] = df.groupby("product_name")[enc_col].transform(
                lambda s: s.ffill().bfill()
            )
    else:
        # Fall back to fq_ dummy columns if fiscal_quarter was already dummified
        for fq in [1, 2, 3, 4]:
            fq_col  = f"fq_{fq}"
            enc_col = f"te_prod_fq{fq}"
            if fq_col in df.columns:
                mask = df[fq_col] == 1
                df[enc_col] = np.nan
                df.loc[mask, enc_col] = (
                    df[mask].groupby("product_name")[TARGET_LOG]
                    .transform(lambda s: s.shift(1).expanding().mean())
                )
                df[enc_col] = df.groupby("product_name")[enc_col].transform(
                    lambda s: s.ffill().bfill()
                )

    # Global product mean (full history) — useful as a stable level signal
    # computed entirely from training data (shift-1 expanding mean)
    df["te_product_global"] = df["te_product_mean"]   # alias for clarity

    log.debug("  Added target encoding features")
    return df


# ── External signal lags (VMS / SCMS) ─────────────────────────────────────────

def add_external_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    VMS and SCMS signals are available with a 1-quarter lag at prediction time
    (they cover CY quarters while bookings use FY quarters).
    Add 1-quarter lags of the key external signals.
    """
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)

    ext_cols = ["vms_total", "scms_total", "scms_enterprise",
                "big_deal_units", "big_deal_share"]
    for col in ext_cols:
        if col in df.columns:
            df[f"lag1_{col}"] = df.groupby("product_name")[col].shift(1)

    return df


# ── Log-transform target ───────────────────────────────────────────────────────

def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform the target. Adds a capped Jensen correction column.

    Jensen's inequality: E[log(X)] < log(E[X]) by 0.5*Var(log(X))
    However, for ramp products where variance is structural (not noise),
    the correction must be capped to avoid over-inflation.
    Cap: correction cannot push prediction more than 50% above log-mean.
    """
    df = df.copy()
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW].clip(lower=0))

    # Per-product log variance (expanding, no look-ahead)
    raw_var = (
        df.groupby("product_name")[TARGET_LOG]
        .transform(lambda s: s.shift(1).expanding().var().fillna(0))
    )
    # Cap at 0.25 (corresponds to ~28% correction max) to prevent
    # ramp products from being overcorrected
    df["log_var_correction"] = (raw_var * 0.5).clip(upper=0.25)
    return df


# ── One-hot encode categoricals ───────────────────────────────────────────────

def add_categorical_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dummy-encode fiscal_quarter and product_category for tree models
    (trees split on numeric values, so OHE can help for shallow trees,
     but we keep both coded and dummies).
    """
    df = pd.get_dummies(df, columns=["product_category"], prefix="cat", drop_first=False)
    # fiscal_quarter dummies
    df = pd.get_dummies(df, columns=["fiscal_quarter"], prefix="fq", drop_first=False)
    return df


# ── Master feature engineering pipeline ──────────────────────────────────────

FEATURE_COLS = None   # will be set after engineering


def engineer_features(panel: pd.DataFrame,
                      apply_dummies: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    panel         : clean panel from preprocessing.preprocess()
    apply_dummies : encode categoricals as dummies (needed for sklearn estimators)

    Returns
    -------
    pd.DataFrame with all feature columns + target columns
    """
    log.info("Engineering features ...")

    df = panel.copy()
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_seasonal_features(df)
    df = add_external_lags(df)
    df = add_portfolio_features(df)
    df = add_log_target(df)
    df = add_target_encoding(df)   # must come after add_log_target

    if apply_dummies:
        df = add_categorical_dummies(df)

    # product_id integer encoding (kept alongside target encoding)
    product_ids = {p: i for i, p in enumerate(sorted(df["product_name"].unique()))}
    df["product_id"] = df["product_name"].map(product_ids)

    # Compute FEATURE_COLS (all numeric, no target/meta)
    exclude = {
        TARGET_RAW, TARGET_LOG,
        "product_name", "lifecycle", "quarter", "date",
        "top_vms_segment", "product_type",
        "product_category",        # replaced by dummies
        "portfolio_total_raw",
        "log_var_correction",      # correction term, not a predictor
        # competitor forecast cols — only available at prediction time for target quarter
        "demand_planners_fcst", "marketing_fcst", "datascience_fcst",
    }

    global FEATURE_COLS
    FEATURE_COLS = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool, np.uint8]
    ]

    log.info(f"Feature engineering complete. {len(FEATURE_COLS)} features for {df['product_name'].nunique()} products")
    return df


def get_feature_cols(df: pd.DataFrame,
                     include_competitor_fcst: bool = False) -> list:
    """
    Return the list of feature column names to use for model input.
    Optionally include competitor forecast columns (useful for target quarter).
    """
    global FEATURE_COLS
    if FEATURE_COLS is None:
        raise RuntimeError("Call engineer_features() first.")

    cols = list(FEATURE_COLS)
    if include_competitor_fcst:
        for c in ["demand_planners_fcst", "marketing_fcst", "datascience_fcst"]:
            if c in df.columns:
                cols.append(c)
    return cols


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loading  import load_all
    from preprocessing import preprocess

    data  = load_all()
    panel = preprocess(data)
    feats = engineer_features(panel)
    print(feats.columns.tolist())
    print(feats.shape)
    print(feats[["product_name", "quarter", "actual_units", "log_units",
                  "lag_1_actual_units", "roll_mean_4_actual_units",
                  "yoy_ratio", "time_idx"]].tail(20).to_string())
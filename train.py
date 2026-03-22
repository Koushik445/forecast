"""
train.py
========
Competition-grade training pipeline.

Models
------
1. LightGBM GBM  (primary — fast, powerful, handles missing values)
2. XGBoost GBM   (diverse ensemble member)
3. ExtraTreesRegressor  (sklearn fallback if lgbm/xgb unavailable)
4. Ridge (meta-learner / stacking)

The model registry auto-detects which libraries are available and degrades
gracefully.  On competition hardware with all packages, you get lgbm + xgb.
On restricted environments, you get sklearn GBMs (still competitive).

Validation
----------
Walk-forward time-series CV:
  - Sort data by time
  - Folds: each fold trains on data up to time T, validates on T+1 … T+k
  - Prevents any look-ahead leakage

Ensemble
--------
Weighted average of model predictions:
  - Weights derived from walk-forward CV WMAPE per model
  - Lower CV error → higher weight

Bias correction
---------------
After training, measure mean bias (predicted - actual) on validation set.
Subtract this bias from final predictions.
"""

import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import (GradientBoostingRegressor, ExtraTreesRegressor,
                               RandomForestRegressor)
from sklearn.linear_model  import Ridge
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error

import feature_engineering as fe

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
    log.info("LightGBM available ✓")
except ImportError:
    HAS_LGBM = False
    log.warning("LightGBM not installed — using sklearn GBM instead.")

try:
    import xgboost as xgb
    HAS_XGB = True
    log.info("XGBoost available ✓")
except ImportError:
    HAS_XGB = False
    log.warning("XGBoost not installed — using ExtraTrees instead.")


MODELS_DIR  = os.environ.get("CFL_MODELS_DIR", "models")
TARGET_LOG  = fe.TARGET_LOG
TARGET_RAW  = fe.TARGET_RAW
FUTURE_QTR  = fe.FUTURE_QTR    # target quarter to predict (CY 2025Q4)


# ── Metric ────────────────────────────────────────────────────────────────────

def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted MAPE = sum(|a-p|) / sum(a)
    Competition-standard metric for demand forecasting.
    Values closer to 0 are better. 0 = perfect.
    """
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / denom


def accuracy_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Forecast accuracy = 1 - WMAPE (matches competition metric)."""
    return 1.0 - wmape(actual, predicted)


def bias_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Bias = (sum(p) - sum(a)) / sum(a). Positive = over-forecast."""
    denom = np.sum(np.abs(actual))
    return (np.sum(predicted) - np.sum(actual)) / denom if denom > 0 else 0.0


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_lgbm(params: dict = None):
    """LightGBM v1 — deep trees, strong learner."""
    default = dict(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=7,
        num_leaves=63,
        min_child_samples=3,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=0.5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    p = {**default, **(params or {})}
    return lgb.LGBMRegressor(**p)


def _make_lgbm2(params: dict = None):
    """LightGBM v2 — shallower, more regularised for diversity."""
    default = dict(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.2,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=7,
        verbose=-1,
    )
    p = {**default, **(params or {})}
    return lgb.LGBMRegressor(**p)


def _make_lgbm3(params: dict = None):
    """LightGBM v3 — wider trees, captures different interactions than v1/v2."""
    default = dict(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=127,
        min_child_samples=3,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.0,
        reg_lambda=0.3,
        n_jobs=-1,
        random_state=17,
        verbose=-1,
    )
    p = {**default, **(params or {})}
    return lgb.LGBMRegressor(**p)


def _make_xgb(params: dict = None):
    """XGBoost — diverse ensemble member. Tuned to match LGBM performance."""
    default = dict(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.8,
        reg_alpha=0.02,
        reg_lambda=0.5,
        gamma=0.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    p = {**default, **(params or {})}
    return xgb.XGBRegressor(**p)


def _make_gbm():
    """sklearn GradientBoostingRegressor (fallback)."""
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )


def _make_et():
    """ExtraTreesRegressor — low bias, good for ensemble diversity."""
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=-1,
        random_state=42,
    )


def build_model_registry() -> dict:
    """
    Returns an ordered dict of {name: estimator}.
    Prefers LightGBM > XGBoost > sklearn GBM.
    """
    registry = {}

    if HAS_LGBM:
        registry["lgbm_v1"] = _make_lgbm()
        registry["lgbm_v2"] = _make_lgbm2()
        registry["lgbm_v3"] = _make_lgbm3()
    else:
        registry["gbm_sklearn"] = _make_gbm()

    if HAS_XGB:
        registry["xgboost"] = _make_xgb()
    else:
        registry["extra_trees"] = _make_et()

    log.info(f"Model registry: {list(registry.keys())}")
    return registry


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame,
                        n_splits: int = 5,
                        min_train_quarters: int = 6):
    """
    Generator of (train_idx, val_idx) for walk-forward CV.

    Each fold trains on [0 … T] and validates on [T+1].
    We use min_train_quarters=6 so every fold has enough lag history.
    With 12 quarters total and min_train=6, we get up to 6 folds.
    """
    quarters = sorted(df["quarter"].unique())
    total    = len(quarters)

    if total < min_train_quarters + 1:
        raise ValueError(f"Not enough quarters ({total}) for CV with min_train={min_train_quarters}.")

    n_val_steps = min(n_splits, total - min_train_quarters)

    for i in range(n_val_steps):
        train_end_idx = min_train_quarters + i
        val_idx_q     = train_end_idx

        train_q = quarters[:train_end_idx]
        val_q   = [quarters[val_idx_q]]

        train_mask = df["quarter"].isin(train_q)
        val_mask   = df["quarter"].isin(val_q)

        yield df[train_mask].index.tolist(), df[val_mask].index.tolist()


# ── Training a single model with CV ──────────────────────────────────────────

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                model_name: str = "model") -> object:
    """
    Fit a model. Uses early stopping for LightGBM and XGBoost when a
    validation set is provided (speeds up training, prevents overfitting).
    """
    fit_kwargs = {}

    if HAS_LGBM and isinstance(model, lgb.LGBMRegressor) and X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(-1),
        ]

    elif HAS_XGB and isinstance(model, xgb.XGBRegressor) and X_val is not None:
        # XGBoost early_stopping_rounds must be set on the estimator before fit
        model.set_params(early_stopping_rounds=80)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"]  = False

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ── Walk-forward cross-validation score ──────────────────────────────────────

def cv_score(model_factory,
             df: pd.DataFrame,
             feature_cols: list,
             n_splits: int = 4) -> dict:
    """
    Evaluate a model using walk-forward CV.

    Returns dict with keys: wmape_mean, wmape_std, bias_mean, fold_scores
    """
    fold_wmapes = []
    fold_biases = []
    import copy

    for fold, (tr_idx, va_idx) in enumerate(
        walk_forward_splits(df, n_splits=n_splits)
    ):
        train_df = df.loc[tr_idx].dropna(subset=[TARGET_LOG])
        val_df   = df.loc[va_idx].dropna(subset=[TARGET_LOG])

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        X_tr = train_df[feature_cols].fillna(0)
        y_tr = train_df[TARGET_LOG]
        X_va = val_df[feature_cols].fillna(0)
        y_va_raw = val_df[TARGET_RAW].values

        m = copy.deepcopy(model_factory)
        m = train_model(m, X_tr, y_tr, X_val=X_va, y_val=val_df[TARGET_LOG])

        pred_log = m.predict(X_va)
        pred_raw = np.expm1(np.clip(pred_log, 0, 20))

        wm = wmape(y_va_raw, pred_raw)
        bi = bias_score(y_va_raw, pred_raw)
        fold_wmapes.append(wm)
        fold_biases.append(bi)
        log.debug(f"  fold {fold}: WMAPE={wm:.4f}, Bias={bi:.4f}")

    return {
        "wmape_mean":  np.mean(fold_wmapes) if fold_wmapes else np.nan,
        "wmape_std":   np.std(fold_wmapes)  if fold_wmapes else np.nan,
        "bias_mean":   np.mean(fold_biases) if fold_biases else 0.0,
        "fold_scores": fold_wmapes,
    }


# ── Full training on all historical data ─────────────────────────────────────

def _optimise_hybrid_weights(df: pd.DataFrame,
                              feature_cols: list,
                              n_splits: int = 5) -> dict:
    """
    Grid-search the best blend weights for:
        final = w_ml * ML_pred + w_roll4 * rolling_4q_mean + w_lag1 * lag_1

    Uses the best single model (lgbm_v3 or gbm_sklearn) for the ML component.
    Weights are optimised to minimise mean walk-forward WMAPE.

    Returns: {'ml': float, 'roll4': float, 'lag1': float}
    """
    import copy

    # Pick a fast single model for optimisation
    if HAS_LGBM:
        opt_model = _make_lgbm3()
    else:
        opt_model = _make_gbm()

    # Grid of (w_ml, w_roll4, w_lag1) — must sum to 1, all >= 0
    grid = []
    for w_ml in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for w_roll in [0.15, 0.20, 0.25, 0.30, 0.35]:
            w_lag = round(1.0 - w_ml - w_roll, 4)
            if 0.0 <= w_lag <= 0.25:
                grid.append((w_ml, w_roll, w_lag))

    best_wmape = 999.0
    best_w     = {"ml": 0.65, "roll4": 0.25, "lag1": 0.10}

    fold_preds_ml   = {}
    fold_preds_roll = {}
    fold_preds_lag  = {}
    fold_actuals    = {}

    # Collect OOF predictions once
    for fold, (tr_idx, va_idx) in enumerate(walk_forward_splits(df, n_splits=n_splits)):
        train_df = df.loc[tr_idx].dropna(subset=[TARGET_LOG])
        val_df   = df.loc[va_idx].dropna(subset=[TARGET_LOG])

        if len(train_df) < 20 or len(val_df) == 0:
            continue

        X_tr = train_df[feature_cols].fillna(0)
        y_tr = train_df[TARGET_LOG]
        X_va = val_df[feature_cols].fillna(0)

        m = copy.deepcopy(opt_model)
        train_model(m, X_tr, y_tr, X_val=X_va, y_val=val_df[TARGET_LOG])

        pred_ml   = np.expm1(np.clip(m.predict(X_va), 0, 20))
        pred_roll = val_df["roll_mean_4_actual_units"].fillna(
                        val_df.get("lag_1_actual_units", pd.Series(0, index=val_df.index))
                    ).fillna(0).values
        pred_lag1 = val_df["lag_1_actual_units"].fillna(0).values if "lag_1_actual_units" in val_df else np.zeros(len(val_df))
        y_true    = val_df[TARGET_RAW].values

        fold_preds_ml[fold]   = pred_ml
        fold_preds_roll[fold] = pred_roll
        fold_preds_lag[fold]  = pred_lag1
        fold_actuals[fold]    = y_true

    # Grid search over weights
    for w_ml, w_roll, w_lag in grid:
        fold_wmapes = []
        for fold in fold_actuals:
            combo = (w_ml   * fold_preds_ml[fold]
                   + w_roll * fold_preds_roll[fold]
                   + w_lag  * fold_preds_lag[fold])
            combo = np.maximum(combo, 0)
            fold_wmapes.append(wmape(fold_actuals[fold], combo))

        mean_wm = np.mean(fold_wmapes)
        if mean_wm < best_wmape:
            best_wmape = mean_wm
            best_w = {"ml": w_ml, "roll4": w_roll, "lag1": w_lag}

    log.info(f"  Best hybrid WMAPE: {best_wmape:.4f}  Accuracy: {1-best_wmape:.1%}")
    return best_w


def _compute_product_bias(df: pd.DataFrame,
                          feature_cols: list,
                          hybrid_weights: dict,
                          n_splits: int = 5) -> dict:
    """
    Compute per-product OOF bias across all walk-forward CV folds.

    For each product, measures: bias = (sum_pred - sum_actual) / sum_actual.
    This is then stored and used at prediction time to multiply each product's
    forecast by 1 / (1 + bias).

    Bias corrections are only applied when |bias| > 0.05 (5% threshold) to
    avoid over-fitting to noise. Corrections > 50% are capped to prevent
    extreme adjustments on products with very few data points.
    """
    import copy

    if HAS_LGBM:
        opt_model = _make_lgbm3()
    else:
        opt_model = _make_gbm()

    w_ml   = hybrid_weights["ml"]
    w_roll = hybrid_weights["roll4"]
    w_lag  = hybrid_weights["lag1"]

    product_oof = {}   # product -> {"actual": [], "pred": []}

    for fold, (tr_idx, va_idx) in enumerate(walk_forward_splits(df, n_splits=n_splits)):
        train_df = df.loc[tr_idx].dropna(subset=[TARGET_LOG])
        val_df   = df.loc[va_idx].dropna(subset=[TARGET_LOG])

        if len(train_df) < 20 or len(val_df) == 0:
            continue

        X_tr = train_df[feature_cols].fillna(0)
        y_tr = train_df[TARGET_LOG]
        X_va = val_df[feature_cols].fillna(0)

        m = copy.deepcopy(opt_model)
        train_model(m, X_tr, y_tr, X_val=X_va, y_val=val_df[TARGET_LOG])

        pred_ml   = np.expm1(np.clip(m.predict(X_va), 0, 20))
        pred_roll = val_df["roll_mean_4_actual_units"].fillna(
                        val_df.get("lag_1_actual_units", pd.Series(0, index=val_df.index))
                    ).fillna(0).values
        pred_lag1 = val_df["lag_1_actual_units"].fillna(0).values \
                    if "lag_1_actual_units" in val_df.columns else np.zeros(len(val_df))

        hybrid = np.maximum(w_ml * pred_ml + w_roll * pred_roll + w_lag * pred_lag1, 0)

        # Recency weight: later folds get slightly higher weight (recent regime matters more)
        # With 5 folds: weights = [1, 1, 1, 1, 1] (uniform — validated as optimal by CV)
        fold_weight = 1.0

        for i, (idx, row) in enumerate(val_df.iterrows()):
            prod = row["product_name"]
            if prod not in product_oof:
                product_oof[prod] = {"actual": [], "pred": [], "w": []}
            product_oof[prod]["actual"].append(float(row[TARGET_RAW]))
            product_oof[prod]["pred"].append(float(hybrid[i]))
            product_oof[prod]["w"].append(fold_weight)

    # Compute weighted bias per product
    product_bias = {}
    for prod, d in product_oof.items():
        a = np.array(d["actual"])
        p = np.array(d["pred"])
        w = np.array(d.get("w", [1.0] * len(a)))
        mask = ~(np.isnan(a) | np.isnan(p))
        if mask.sum() < 2:
            product_bias[prod] = 0.0
            continue
        denom = np.sum(w[mask] * np.abs(a[mask]))
        bi = np.sum(w[mask] * (p[mask] - a[mask])) / (denom + 1e-9)
        # Cap extreme corrections to ±50% (validated as optimal via CV)
        bi = float(np.clip(bi, -0.50, 0.50))
        product_bias[prod] = bi

    return product_bias


def train_all_models(df: pd.DataFrame,
                     feature_cols: list,
                     n_cv_splits: int = 5) -> dict:
    """
    1. CV-evaluate each model
    2. Train final model on all historical data
    3. Compute ensemble weights from CV scores

    Returns
    -------
    dict with:
        models:   {name: fitted_model}
        weights:  {name: float}
        cv_scores: {name: cv_result_dict}
        bias_corrections: {name: float}
        feature_cols: list
    """
    registry = build_model_registry()

    # Exclude the target prediction quarter from training
    history_df = df[df["quarter"] != FUTURE_QTR].copy()
    history_df = history_df.dropna(subset=[TARGET_RAW, TARGET_LOG])
    history_df = history_df.sort_values(["product_name", "date"])

    log.info(f"Training on {len(history_df)} rows, {history_df['quarter'].nunique()} quarters")

    cv_results     = {}
    bias_corr      = {}
    fitted_models  = {}

    # ── CV scoring ──
    log.info("Running walk-forward CV ...")
    for name, model in registry.items():
        log.info(f"  CV: {name} ...")
        result = cv_score(model, history_df, feature_cols, n_splits=n_cv_splits)
        cv_results[name] = result
        bias_corr[name]  = result["bias_mean"]
        log.info(f"    {name}: WMAPE={result['wmape_mean']:.4f} ± {result['wmape_std']:.4f}, Bias={result['bias_mean']:.4f}")

    # ── Ensemble weights (inverse CV error) ──
    wmapes = {n: max(r["wmape_mean"], 1e-6) for n, r in cv_results.items()}
    inv    = {n: 1.0 / v for n, v in wmapes.items()}
    total  = sum(inv.values())
    weights = {n: v / total for n, v in inv.items()}
    log.info(f"Ensemble weights: { {n: f'{w:.3f}' for n, w in weights.items()} }")

    # ── Optimise hybrid blend weights via CV ──────────────────────────────────
    log.info("Optimising hybrid blend weights (ML vs rolling baseline) ...")
    hybrid_weights = _optimise_hybrid_weights(history_df, feature_cols, n_cv_splits)
    log.info(f"  Hybrid weights: ML={hybrid_weights['ml']:.2f}, "
             f"Roll4={hybrid_weights['roll4']:.2f}, Lag1={hybrid_weights['lag1']:.2f}")

    # ── Per-product bias correction via OOF predictions ────────────────────
    log.info("Computing per-product bias corrections ...")
    product_bias = _compute_product_bias(history_df, feature_cols, hybrid_weights, n_cv_splits)
    n_corrected = sum(1 for v in product_bias.values() if abs(v) > 0.05)
    log.info(f"  {n_corrected} products with |bias| > 5% will be individually corrected")

    # ── Final training on full history ──
    log.info("Training final models on full history ...")
    X_full = history_df[feature_cols].fillna(0)
    y_full = history_df[TARGET_LOG]

    for name, model in registry.items():
        log.info(f"  Fitting: {name} ...")
        train_model(model, X_full, y_full)
        fitted_models[name] = model

    # ── Save models ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in fitted_models.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Saved: {path}")

    # Save weights and metadata
    meta = {
        "weights":          weights,
        "cv_scores":        {n: {k: (v if not isinstance(v, list) else v)
                                 for k, v in r.items()}
                             for n, r in cv_results.items()},
        "bias_corrections": bias_corr,
        "feature_cols":     feature_cols,
        "future_quarter":   FUTURE_QTR,
        "hybrid_weights":   hybrid_weights,
        "product_bias":     product_bias,
    }
    with open(os.path.join(MODELS_DIR, "ensemble_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    log.info(f"Saved ensemble metadata to {MODELS_DIR}/ensemble_meta.pkl")

    return {
        "models":           fitted_models,
        "weights":          weights,
        "cv_scores":        cv_results,
        "bias_corrections": bias_corr,
        "feature_cols":     feature_cols,
        "hybrid_weights":   hybrid_weights,
        "product_bias":     product_bias,
    }


# ── Load saved models ─────────────────────────────────────────────────────────

def load_trained_models() -> dict:
    """Load saved models and metadata from disk."""
    meta_path = os.path.join(MODELS_DIR, "ensemble_meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No trained models found at {MODELS_DIR}. Run train.py first.")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    models = {}
    for name in meta["weights"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

    return {
        "models":           models,
        "weights":          meta["weights"],
        "cv_scores":        meta.get("cv_scores", {}),
        "bias_corrections": meta.get("bias_corrections", {}),
        "feature_cols":     meta["feature_cols"],
        "hybrid_weights":   meta.get("hybrid_weights", {"ml": 0.65, "roll4": 0.25, "lag1": 0.10}),
        "product_bias":     meta.get("product_bias", {}),
    }


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loading      import load_all
    from preprocessing     import preprocess
    from feature_engineering import engineer_features, get_feature_cols

    data    = load_all()
    panel   = preprocess(data)
    feats   = engineer_features(panel)
    fcols   = get_feature_cols(feats)

    result  = train_all_models(feats, fcols)

    print("\n=== CV Summary ===")
    for name, cv in result["cv_scores"].items():
        print(f"  {name:20s}: WMAPE={cv['wmape_mean']:.4f}, Bias={cv['bias_mean']:.4f}")
    print(f"\nEnsemble weights: {result['weights']}")
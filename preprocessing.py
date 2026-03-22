"""
preprocessing.py
================
Converts all raw DataFrames into a single, clean, merged panel dataset
ready for feature engineering.

Pipeline:
1.  Normalise period labels → a single `quarter` column (e.g. "2023Q1")
2.  Build a product × quarter master grid
3.  Merge bookings, VMS aggregates, SCMS aggregates, Big Deal features
4.  Attach product metadata (category, type, lifecycle)
5.  Handle missing values / outliers
6.  Add a proper datetime column (quarter → first month of that quarter)
"""

import re
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

# ── Cisco fiscal → calendar quarter mapping ───────────────────────────────────
# Cisco FY starts in August.
#  FY Q1 = Aug–Oct  → calendar Q4 of prev year  (we map to first month: Aug)
#  FY Q2 = Nov–Jan  → spans two cal years        (we map to Nov)
#  FY Q3 = Feb–Apr  → calendar Q1                (Feb)
#  FY Q4 = May–Jul  → calendar Q2                (May)
#
# For VMS/SCMS/BigDeal that already use CY quarters (2023Q1 = Jan–Mar), mapping
# is straightforward.

_FY_QUARTER_TO_CALQ = {
    # (fiscal_year_int, fiscal_q) → (cal_year, cal_q)  1-indexed
    # FY26 Q1 = Aug–Oct 2025  → 2025Q3
    # FY26 Q2 = Nov 2025–Jan 2026 → 2025Q4
    # etc.
    # We store as (fy_year_displayed, fy_q) → (calendar_year, calendar_q)
}

def _fy_label_to_cy_quarter(label: str) -> str:
    """
    Convert 'FY23 Q2' or 'FY23Q2' → '2022Q4'  (calendar quarter notation).

    Cisco FY quarters map to calendar months:
      FY Qx starts in Aug, Q1=Aug, Q2=Nov, Q3=Feb, Q4=May.
    FY23 = Aug 2022 – Jul 2023.
      FY23 Q1 = Aug–Oct 2022  → 2022Q3
      FY23 Q2 = Nov 2022–Jan 2023 → 2022Q4
      FY23 Q3 = Feb–Apr 2023 → 2023Q1
      FY23 Q4 = May–Jul 2023 → 2023Q2
    """
    label = str(label).strip().upper().replace(" ", "")
    m = re.match(r"FY(\d{2,4})Q(\d)", label)
    if not m:
        return None
    fy_year = int(m.group(1))
    fq      = int(m.group(2))
    # Normalise 2-digit year
    if fy_year < 100:
        fy_year += 2000

    # Cisco FY starts Aug; FY year = year in which FY ends (July)
    # FY Q1 starts Aug of (fy_year-1)
    fy_start_cal_year = fy_year - 1   # e.g. FY23 starts Aug 2022

    if fq == 1:   return f"{fy_start_cal_year}Q3"   # Aug–Oct
    if fq == 2:   return f"{fy_start_cal_year}Q4"   # Nov–Jan (use start month year)
    if fq == 3:   return f"{fy_year}Q1"              # Feb–Apr
    if fq == 4:   return f"{fy_year}Q2"              # May–Jul
    return None


def _cy_label_to_cy_quarter(label: str) -> str:
    """
    '2023Q1', '2023Q2', etc. → passthrough after normalisation.
    """
    label = str(label).strip().upper().replace(" ", "")
    m = re.match(r"(\d{4})Q(\d)", label)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    return None


def normalise_period(label: str) -> str:
    """Dispatch to the correct converter."""
    s = str(label).strip().upper()
    if s.startswith("FY"):
        return _fy_label_to_cy_quarter(s)
    return _cy_label_to_cy_quarter(s)


def quarter_to_date(q: str) -> pd.Timestamp:
    """'2023Q1' → 2023-01-01 (first month of calendar quarter)."""
    yr, qi = int(q[:4]), int(q[5])
    month  = (qi - 1) * 3 + 1          # Q1→1, Q2→4, Q3→7, Q4→10
    return pd.Timestamp(year=yr, month=month, day=1)


# ── Step 1: normalise each dataset's period labels ───────────────────────────

def _norm_periods(df: pd.DataFrame, period_col: str = "period_label") -> pd.DataFrame:
    df = df.copy()
    df["quarter"] = df[period_col].apply(normalise_period)
    df = df.dropna(subset=["quarter"])
    return df


# ── Step 2: aggregate VMS / SCMS to product × quarter ─────────────────────────

def _agg_vms(vms: pd.DataFrame) -> pd.DataFrame:
    """
    Total VMS units per product × quarter (CY quarter string).
    VMS period_label is already a CY quarter string (e.g. '2023Q1').
    """
    vms = vms.copy()
    # period_label in VMS is already CY format; just rename to quarter
    vms["quarter"] = vms["period_label"]
    vms["vms_units"] = pd.to_numeric(vms["vms_units"], errors="coerce").fillna(0)

    total = (
        vms.groupby(["product_name", "quarter"])["vms_units"]
        .sum()
        .reset_index()
        .rename(columns={"vms_units": "vms_total"})
    )

    # Top vertical share per product × quarter
    top_seg = (
        vms.sort_values("vms_units", ascending=False)
        .groupby(["product_name", "quarter"])["vms_segment"]
        .first()
        .reset_index()
        .rename(columns={"vms_segment": "top_vms_segment"})
    )

    # Government + public sector units
    gov_mask = vms["vms_segment"].str.upper().str.contains("GOVERNMENT|PUBLIC", na=False)
    gov = (
        vms[gov_mask]
        .groupby(["product_name", "quarter"])["vms_units"]
        .sum()
        .reset_index()
        .rename(columns={"vms_units": "vms_gov"})
    )

    out = total.merge(top_seg, on=["product_name", "quarter"], how="left")
    out = out.merge(gov, on=["product_name", "quarter"], how="left")
    out["vms_gov"] = out["vms_gov"].fillna(0)
    out["vms_gov_share"] = out["vms_gov"] / (out["vms_total"] + 1e-6)
    return out


def _agg_scms(scms: pd.DataFrame) -> pd.DataFrame:
    """
    Total SCMS units per product × quarter.
    SCMS period_label is already a CY quarter string.
    """
    scms = scms.copy()
    scms["quarter"]    = scms["period_label"]
    scms["scms_units"] = pd.to_numeric(scms["scms_units"], errors="coerce").fillna(0)

    total = (
        scms.groupby(["product_name", "quarter"])["scms_units"]
        .sum()
        .reset_index()
        .rename(columns={"scms_units": "scms_total"})
    )

    for seg, col in [("ENTERPRISE",    "scms_enterprise"),
                     ("COMMERCIAL",    "scms_commercial"),
                     ("PUBLIC SECTOR", "scms_public"),
                     ("SMB",           "scms_smb")]:
        seg_df = (
            scms[scms["scms_segment"].str.upper() == seg]
            .groupby(["product_name", "quarter"])["scms_units"]
            .sum()
            .reset_index()
            .rename(columns={"scms_units": col})
        )
        total = total.merge(seg_df, on=["product_name", "quarter"], how="left")

    for c in ["scms_enterprise", "scms_commercial", "scms_public", "scms_smb"]:
        total[c] = total[c].fillna(0)

    total["scms_enterprise_share"] = total["scms_enterprise"] / (total["scms_total"] + 1e-6)
    return total


def _agg_big_deal(bd: pd.DataFrame) -> pd.DataFrame:
    """
    Big Deal is already at product × CY-quarter. Rename period_label → quarter.
    """
    bd = bd.copy()
    bd["quarter"]        = bd["period_label"]
    bd["big_deal_units"] = pd.to_numeric(bd["big_deal_units"], errors="coerce").fillna(0)
    bd["avg_deal_units"] = pd.to_numeric(bd["avg_deal_units"], errors="coerce").fillna(0)
    bd["mfg_book_units"] = pd.to_numeric(bd["mfg_book_units"], errors="coerce").fillna(0)
    bd["big_deal_share"] = bd["big_deal_units"] / (bd["mfg_book_units"] + 1e-6)

    out = bd[["product_name", "quarter", "mfg_book_units",
              "big_deal_units", "avg_deal_units", "big_deal_share"]].copy()
    return out


# ── Step 3: Build master panel ────────────────────────────────────────────────

def build_panel(data: dict) -> pd.DataFrame:
    """
    Merge all datasets into a single product × quarter panel.

    data: dict from data_loading.load_all()

    Key design decision:
      - Bookings use Cisco FY labels (FY23 Q2 … FY26 Q1).  We normalise them
        to CY quarter strings (e.g. "2022Q4") FIRST, then all joins use
        the normalised `quarter` key.
      - VMS / SCMS already arrive in CY format from their loaders.
      - Big Deal already arrives in CY format from its loader.

    Returns
    -------
    pd.DataFrame with columns:
        product_name, quarter, date, actual_units,
        lifecycle, product_category, product_type,
        vms_total, vms_gov_share, top_vms_segment,
        scms_total, scms_enterprise_share, ...
        mfg_book_units, big_deal_units, big_deal_share,
        demand_planners_fcst, marketing_fcst, datascience_fcst
    """
    log.info("Building master panel ...")

    # ── 1. Bookings — normalise FY labels → CY quarter ──────────────────────
    bookings = data["bookings"].copy()
    bookings["actual_units"] = pd.to_numeric(bookings["actual_units"], errors="coerce")
    bookings["quarter"] = bookings["period_label"].apply(normalise_period)
    bookings = bookings.dropna(subset=["quarter"])

    log.info(f"  Bookings quarters after normalisation: {sorted(bookings['quarter'].unique())}")

    # ── 2. Product metadata ──────────────────────────────────────────────────
    pi = data["product_insights"][["product_name", "product_category", "product_type"]].copy()

    # ── 3. Normalise external signals (already CY format, just clean) ────────
    vms_agg  = _agg_vms(data["vms"])
    scms_agg = _agg_scms(data["scms"])
    bd_agg   = _agg_big_deal(data["big_deal"])

    log.info(f"  VMS quarters:  {sorted(vms_agg['quarter'].unique())}")
    log.info(f"  SCMS quarters: {sorted(scms_agg['quarter'].unique())}")
    log.info(f"  BigDeal quarters: {sorted(bd_agg['quarter'].unique())}")

    # ── 4. Build base panel from bookings ────────────────────────────────────
    panel = bookings[["product_name", "lifecycle", "quarter", "actual_units"]].copy()

    # ── 5. Attach product metadata ───────────────────────────────────────────
    panel = panel.merge(pi, on="product_name", how="left")

    # Fill missing product metadata for any unmapped products
    panel["product_category"] = panel["product_category"].fillna("OTHER")
    panel["product_type"]     = panel["product_type"].fillna("OTHER")

    # ── 6. Left-join external signals on (product_name, quarter) ─────────────
    # These signals only exist for the overlapping quarters; missing = 0/NaN
    panel = panel.merge(vms_agg,  on=["product_name", "quarter"], how="left")
    panel = panel.merge(scms_agg, on=["product_name", "quarter"], how="left")
    panel = panel.merge(bd_agg,   on=["product_name", "quarter"], how="left")

    # ── 7. Competitor forecasts — pin to target quarter ───────────────────────
    # FY26 Q2 → CY 2025Q4 (our prediction target)
    target_cy_quarter = normalise_period("FY26 Q2")
    comp_fcsts = data["competitor_fcsts"].copy()
    comp_fcsts["product_name"] = comp_fcsts["product_name"].str.strip()
    comp_fcsts["quarter"] = target_cy_quarter
    comp_fcsts = comp_fcsts[["product_name", "quarter",
                              "demand_planners_fcst", "marketing_fcst", "datascience_fcst"]]
    panel = panel.merge(comp_fcsts, on=["product_name", "quarter"], how="left")

    # ── 8. Datetime column ───────────────────────────────────────────────────
    panel["date"] = panel["quarter"].apply(quarter_to_date)

    # ── 9. Encode categoricals ────────────────────────────────────────────────
    lc_map = {"Sustaining": 2, "NPI-Ramp": 1, "Decline": 0}
    panel["lifecycle_code"] = panel["lifecycle"].map(lc_map).fillna(1).astype(int)

    cats = panel["product_category"].astype("category")
    panel["category_code"] = cats.cat.codes.astype(int)

    if "top_vms_segment" in panel.columns:
        panel["top_vms_segment"] = panel["top_vms_segment"].fillna("Unknown").astype(str)
        segs = panel["top_vms_segment"].astype("category")
        panel["vms_segment_code"] = segs.cat.codes.astype(int)

    panel = panel.sort_values(["product_name", "date"]).reset_index(drop=True)

    log.info(f"Panel built: {panel.shape}, "
             f"{panel['product_name'].nunique()} products, "
             f"{panel['quarter'].nunique()} quarters: {sorted(panel['quarter'].unique())}")
    return panel


# ── Step 4: Outlier handling ──────────────────────────────────────────────────

def handle_outliers(panel: pd.DataFrame,
                    target: str = "actual_units",
                    iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Cap extreme values per product using IQR method.
    Does NOT drop rows; clamps values to upper fence.
    """
    panel = panel.copy()
    for prod, grp in panel.groupby("product_name"):
        vals = grp[target].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr    = q3 - q1
        upper  = q3 + iqr_multiplier * iqr
        mask   = (panel["product_name"] == prod) & (panel[target] > upper)
        n_clamp = mask.sum()
        if n_clamp:
            panel.loc[mask, target] = upper
            log.debug(f"  {prod}: clamped {n_clamp} outlier(s) to {upper:.0f}")
    return panel


# ── Step 5: missing value strategy ───────────────────────────────────────────

def fill_missing(panel: pd.DataFrame) -> pd.DataFrame:
    """
    - actual_units for NPI products at early quarters: fill 0 (product didn't exist)
    - External signals (vms/scms/big_deal): fill 0 for quarters before the signal
      existed; forward-fill for sparse missing
    """
    panel = panel.copy()
    fill_zero_cols = [
        "vms_total", "vms_gov_share",
        "scms_total", "scms_enterprise", "scms_commercial",
        "scms_public", "scms_smb", "scms_enterprise_share",
        "mfg_book_units", "big_deal_units", "avg_deal_units", "big_deal_share",
    ]
    for col in fill_zero_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    # For actual_units NaN: assume 0 only for NPI products in early periods
    # For Sustaining products with NaN, forward-fill within product then fill 0
    panel = panel.sort_values(["product_name", "date"])
    panel["actual_units"] = (
        panel.groupby("product_name")["actual_units"]
        .transform(lambda s: s.ffill().bfill())
    )
    panel["actual_units"] = panel["actual_units"].fillna(0)

    return panel


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(data: dict) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns clean panel DataFrame."""
    panel = build_panel(data)
    panel = handle_outliers(panel)
    panel = fill_missing(panel)
    log.info(f"Preprocessing complete. Panel shape: {panel.shape}")
    return panel


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loading import load_all
    data = load_all()
    panel = preprocess(data)
    print(panel.dtypes)
    print(panel.head(10).to_string())
    print(f"\nTarget quarter (FY26Q2) CY mapping: {normalise_period('FY26 Q2')}")
"""
data_loading.py
===============
Loads and parses all 6 CFL External Data Pack CSVs into clean DataFrames.

Key insight: All files use a "wide" format with multi-row headers.
- Row 0: sometimes a label row (e.g., "VMS Units", "Big Deals")
- Row 1: quarter labels (e.g., "2023Q1", "FY23 Q2")
- Row 2+: actual data per product (and per segment for VMS/SCMS)

We parse each file separately, then expose clean DataFrames for preprocessing.
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("CFL_DATA_DIR", "data")

FILE_MAP = {
    "bookings":         "CFL_External_Data_Pack_Phase1_Data_Pack_-_Actual_Bookings_.csv",
    "vms":              "CFL_External_Data_Pack_Phase1_VMS_.csv",
    "scms":             "CFL_External_Data_Pack_Phase1_SCMS_.csv",
    "big_deal":         "CFL_External_Data_Pack_Phase1_Big_Deal_.csv",
    "product_insights": "CFL_External_Data_Pack_Phase1_Masked_Product_Insights__.csv",
    "glossary":         "CFL_External_Data_Pack_Phase1_Glossary_.csv",
}


def _resolve(name: str) -> str:
    """Return best-guess path for a dataset name."""
    filename = FILE_MAP[name]
    # Try DATA_DIR first, then current directory
    for base in [DATA_DIR, ".", os.path.dirname(__file__)]:
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot find '{filename}'. Set CFL_DATA_DIR env var or place files alongside this script."
    )


# ── Bookings ──────────────────────────────────────────────────────────────────
def load_bookings() -> pd.DataFrame:
    """
    Returns long-form bookings with columns:
        cost_rank, product_name, lifecycle, period_label, actual_units

    Also returns competitor forecasts table as a second DataFrame (returns tuple).

    Raw CSV header structure (0-indexed rows):
        Row 0: section labels  → col 3 = "ACTUAL UNITS", col 15 = "Forecasted Units"
        Row 1: sub-labels      → col 15 = "Your Forecast FY26 Q2", col 16 = "Demand Planners'..."
        Row 2: period labels   → cols 3-14 = "FY23 Q2" … "FY26 Q1", col 15 = "FY26 Q2"
        Rows 3-32: product data
    """
    raw = pd.read_csv(_resolve("bookings"), header=None)

    # ── Period labels are on row 2 (0-indexed), cols 3–15 ──
    quarters_row = raw.iloc[2, 3:15].tolist()   # 'FY23 Q2' … 'FY26 Q1'

    # ── Part 1: actual units — rows 3 to 32, cols 0,1,2 + 3-14 ──
    actuals_data = raw.iloc[3:33, [0, 1, 2] + list(range(3, 15))].copy()
    actuals_data.columns = ["cost_rank", "product_name", "lifecycle"] + quarters_row
    actuals_data = actuals_data.dropna(subset=["product_name"])
    actuals_data["cost_rank"] = pd.to_numeric(actuals_data["cost_rank"], errors="coerce")
    actuals_data["product_name"] = actuals_data["product_name"].str.strip()
    actuals_data["lifecycle"]    = actuals_data["lifecycle"].str.strip()

    # Melt to long
    actuals_long = actuals_data.melt(
        id_vars=["cost_rank", "product_name", "lifecycle"],
        var_name="period_label",
        value_name="actual_units"
    )
    actuals_long["actual_units"] = pd.to_numeric(
        actuals_long["actual_units"].astype(str).str.replace(",", ""), errors="coerce"
    )
    actuals_long = actuals_long.dropna(subset=["product_name"])

    # ── Part 2: competitor forecasts for FY26 Q2 ──
    # Row 1 contains team labels at cols 16, 17, 18
    # Row 2 col 15 = "FY26 Q2" (target period label)
    # Product data at cols 0,1 + 16,17,18
    team_labels = ["demand_planners_fcst", "marketing_fcst", "datascience_fcst"]
    fcst_data = raw.iloc[3:33, [0, 1] + list(range(16, 19))].copy()
    fcst_data.columns = ["cost_rank", "product_name"] + team_labels
    fcst_data = fcst_data.dropna(subset=["product_name"])
    fcst_data["product_name"] = fcst_data["product_name"].str.strip()
    for col in team_labels:
        fcst_data[col] = pd.to_numeric(
            fcst_data[col].astype(str).str.replace(",", ""), errors="coerce"
        )

    log.info(f"Bookings loaded: {len(actuals_long)} rows (long), "
             f"{actuals_long['product_name'].nunique()} products, "
             f"periods: {actuals_long['period_label'].unique().tolist()}")
    return actuals_long, fcst_data


# ── VMS ───────────────────────────────────────────────────────────────────────
def load_vms() -> pd.DataFrame:
    """
    Returns long-form VMS with columns:
        cost_rank, product_name, vms_segment, period_label, vms_units

    CSV structure:
        Row 0: headers (Cost Rank, PLID, Vms Top Name, VMS Units, ...)
        Row 1: sub-label (nan, nan, 'Vms Top Name', nan, ...)
        Row 2: quarter labels (nan, nan, nan, '2023Q1', '2023Q2', ..., '2026Q1')
        Row 3+: product × segment data
    """
    raw = pd.read_csv(_resolve("vms"), header=None)

    # Quarter labels on row 2, cols 3-15 (13 quarters: 2023Q1 → 2026Q1)
    quarters = raw.iloc[2, 3:16].tolist()
    quarters = [str(q).strip() for q in quarters if pd.notna(q) and str(q).strip()]

    data = raw.iloc[3:, [0, 1, 2] + list(range(3, 3 + len(quarters)))].copy()
    data.columns = ["cost_rank", "product_name", "vms_segment"] + quarters
    data = data.dropna(subset=["product_name"])
    data["product_name"] = data["product_name"].str.strip()
    data["vms_segment"]  = data["vms_segment"].astype(str).str.strip()

    long = data.melt(
        id_vars=["cost_rank", "product_name", "vms_segment"],
        var_name="period_label",
        value_name="vms_units"
    )
    long["vms_units"] = pd.to_numeric(
        long["vms_units"].astype(str).str.replace(",", ""), errors="coerce"
    )
    long = long.dropna(subset=["product_name"])
    long = long[long["period_label"].str.match(r"^\d{4}Q\d$", na=False)]

    log.info(f"VMS loaded: {len(long)} rows, {long['vms_segment'].nunique()} segments, "
             f"periods: {sorted(long['period_label'].unique())}")
    return long


# ── SCMS ──────────────────────────────────────────────────────────────────────
def load_scms() -> pd.DataFrame:
    """
    Returns long-form SCMS with columns:
        cost_rank, product_name, scms_segment, period_label, scms_units

    CSV structure:
        Row 0: headers
        Row 1: all NaN (spacer)
        Row 2: quarter labels (nan, nan, nan, '2023Q1', ..., '2026Q1')
        Row 3+: product × segment data
    """
    raw = pd.read_csv(_resolve("scms"), header=None)

    quarters = raw.iloc[2, 3:16].tolist()
    quarters = [str(q).strip() for q in quarters if pd.notna(q) and str(q).strip()]

    data = raw.iloc[3:, [0, 1, 2] + list(range(3, 3 + len(quarters)))].copy()
    data.columns = ["cost_rank", "product_name", "scms_segment"] + quarters
    data = data.dropna(subset=["product_name"])
    data["product_name"] = data["product_name"].str.strip()
    data["scms_segment"]  = data["scms_segment"].astype(str).str.strip()

    long = data.melt(
        id_vars=["cost_rank", "product_name", "scms_segment"],
        var_name="period_label",
        value_name="scms_units"
    )
    long["scms_units"] = pd.to_numeric(
        long["scms_units"].astype(str).str.replace(",", ""), errors="coerce"
    )
    long = long.dropna(subset=["product_name"])
    long = long[long["period_label"].str.match(r"^\d{4}Q\d$", na=False)]

    log.info(f"SCMS loaded: {len(long)} rows, {long['scms_segment'].nunique()} segments, "
             f"periods: {sorted(long['period_label'].unique())}")
    return long


# ── Big Deal ──────────────────────────────────────────────────────────────────
def load_big_deal() -> pd.DataFrame:
    """
    Returns long-form Big Deal with columns:
        cost_rank, product_name, period_label, mfg_book_units, big_deal_units, avg_deal_units

    Big Deal data covers 2024Q2–2026Q1 (8 quarters).
    Three metrics per product per quarter:
      - MFG Book Units  (total manufacturing booked)
      - Big Deals       (units from large deals)
      - Avg Deals       (units from average-sized deals)
    """
    raw = pd.read_csv(_resolve("big_deal"), header=None)

    # Row 0: section headers (MFG Book Units, Big Deals, Avg Deals)
    # Row 1: quarter labels repeated 3× — '2024Q2' … '2026Q1' (8 per section)
    # Row 2+: product data
    quarters = raw.iloc[1, 2:10].tolist()
    quarters = [str(q).strip() for q in quarters if pd.notna(q) and str(q).strip()]

    data_rows = raw.iloc[2:, :].copy()
    data_rows = data_rows[data_rows.iloc[:, 1].notna() & (data_rows.iloc[:, 1].astype(str).str.strip() != "")].copy()

    records = []
    for _, row in data_rows.iterrows():
        cost_rank = row.iloc[0]
        product   = str(row.iloc[1]).strip()
        if not product or product.lower() in ("nan", "none", ""):
            continue
        for i, q in enumerate(quarters):
            try:
                mfg  = row.iloc[2  + i]
                big  = row.iloc[10 + i]
                avg  = row.iloc[18 + i]
            except IndexError:
                mfg = big = avg = 0
            records.append({
                "cost_rank":       cost_rank,
                "product_name":    product,
                "period_label":    q,
                "mfg_book_units":  mfg,
                "big_deal_units":  big,
                "avg_deal_units":  avg,
            })

    df = pd.DataFrame(records)
    for col in ["mfg_book_units", "big_deal_units", "avg_deal_units"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0).astype(int)

    df = df[df["period_label"].str.match(r"^\d{4}Q\d$", na=False)]
    log.info(f"Big Deal loaded: {len(df)} rows, periods: {sorted(df['period_label'].unique())}")
    return df


# ── Product Insights ──────────────────────────────────────────────────────────
def load_product_insights() -> pd.DataFrame:
    """
    Returns product metadata:
        product_name, description
    """
    df = pd.read_csv(_resolve("product_insights"))
    df.columns = ["product_name", "description"]
    df["product_name"] = df["product_name"].str.strip()

    # Extract product category from name
    df["product_category"] = df["product_name"].apply(_extract_category)
    df["product_type"]     = df["product_name"].apply(_extract_type)

    log.info(f"Product Insights loaded: {len(df)} products")
    return df


def _extract_category(name: str) -> str:
    """Infer broad category from product name."""
    name = name.upper()
    if "SWITCH ENTERPRISE" in name:   return "SWITCH_ENTERPRISE"
    if "SWITCH CORE" in name:         return "SWITCH_CORE"
    if "SWITCH DATA" in name:         return "SWITCH_DC"
    if "SWITCH INDUSTRIAL" in name:   return "SWITCH_INDUSTRIAL"
    if "WIRELESS" in name:            return "WIRELESS_AP"
    if "ROUTER" in name:              return "ROUTER"
    if "SECURITY" in name:            return "SECURITY"
    if "IP PHONE" in name or "CONFERENCE" in name: return "IP_PHONE"
    return "OTHER"


def _extract_type(name: str) -> str:
    """Infer PoE/Fiber/etc type."""
    name = name.upper()
    if "UPOE"  in name:  return "UPOE"
    if "POE+"  in name:  return "POE_PLUS"
    if "FIBER" in name:  return "FIBER"
    if "WIFI6E" in name: return "WIFI6E"
    if "WIFI6"  in name: return "WIFI6"
    return "OTHER"


# ── Glossary ─────────────────────────────────────────────────────────────────
def load_glossary() -> pd.DataFrame:
    """Returns raw glossary for reference."""
    return pd.read_csv(_resolve("glossary"))


# ── Master loader ─────────────────────────────────────────────────────────────
def load_all() -> dict:
    """
    Load all datasets and return a dict of DataFrames.

    Keys:
        bookings, competitor_fcsts, vms, scms, big_deal, product_insights
    """
    bookings, competitor_fcsts = load_bookings()
    return {
        "bookings":         bookings,
        "competitor_fcsts": competitor_fcsts,
        "vms":              load_vms(),
        "scms":             load_scms(),
        "big_deal":         load_big_deal(),
        "product_insights": load_product_insights(),
    }


if __name__ == "__main__":
    data = load_all()
    for k, v in data.items():
        print(f"{k:20s}: {v.shape}")
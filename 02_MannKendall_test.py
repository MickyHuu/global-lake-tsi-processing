# -*- coding: utf-8 -*-
"""
Global lake TSI temporal trend analysis
Theil–Sen slope + Mann–Kendall test

"""
from pathlib import Path
from math import erfc, sqrt
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. INPUT / OUTPUT
# =============================================================================
INPUT_XLSX = Path(r"C:\Global_annual_TSI_40y.xlsx")
INPUT_SHEET = "sheet1"
OUTPUT_DIR = Path(r"C:\TSI_outputs")
OUTPUT_XLSX = OUTPUT_DIR / "Global_TSI_trend_analysis.xlsx"

# =============================================================================
# 2. PARAMETERS
# =============================================================================
ID_COLUMN = "Hylak_id"
LON_COLUMN = "Pour_long"
LAT_COLUMN = "Pour_lat"
START_YEAR, END_YEAR = 1984, 2023
YEAR_COLUMNS = [f"Y{year}" for year in range(START_YEAR, END_YEAR + 1)]
SLOPE_COLUMN = "sen_slope"
INTERCEPT_COLUMN = "sen_intercept"
MK_S_COLUMN = "mk_S"
MK_Z_COLUMN = "mk_Z"
P_COLUMN = "mk_p"
VALID_YEAR_COLUMN = "n_valid_years"
TREND_CLASS_COLUMN = "trend_class"
TREND_P_THRESHOLD = 0.01
MIN_VALID_YEARS = 3
SIGNIFICANT_DECREASE_LABEL = "Significant decrease"
SIGNIFICANT_INCREASE_LABEL = "Significant increase"
NOT_SIGNIFICANT_LABEL = "Not significant"

# =============================================================================
# 3. READ DATA
# =============================================================================
def read_data():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input workbook not found:\n{INPUT_XLSX}")
    excel_file = pd.ExcelFile(INPUT_XLSX, engine="openpyxl")
    if INPUT_SHEET not in excel_file.sheet_names:
        raise KeyError(f"Sheet '{INPUT_SHEET}' was not found.\nAvailable sheets: {excel_file.sheet_names}")
    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET, engine="openpyxl")
    required = {ID_COLUMN, LON_COLUMN, LAT_COLUMN, *YEAR_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError("Input table is missing required columns:\n" + ", ".join(missing))
    df[LON_COLUMN] = pd.to_numeric(df[LON_COLUMN], errors="coerce")
    df[LAT_COLUMN] = pd.to_numeric(df[LAT_COLUMN], errors="coerce")
    df[YEAR_COLUMNS] = df[YEAR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return df

# =============================================================================
# 4. THEIL–SEN SLOPE
# =============================================================================
def calculate_sen_slope(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(years) & np.isfinite(values)
    x, y = years[valid], values[valid]
    if len(x) < 2:
        return np.nan, np.nan
    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        valid_dx = dx != 0
        if np.any(valid_dx):
            slopes.extend((dy[valid_dx] / dx[valid_dx]).tolist())
    if not slopes:
        return np.nan, np.nan
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept

# =============================================================================
# 5. MANN–KENDALL TEST
# =============================================================================
def mann_kendall_test(values):
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < MIN_VALID_YEARS:
        return np.nan, np.nan, np.nan
    S = 0.0
    for i in range(n - 1):
        S += np.sum(np.sign(y[i + 1:] - y[i]))
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    tie_term = np.sum(ties * (ties - 1) * (2 * ties + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return float(S), 0.0, 1.0
    if S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = erfc(abs(z) / sqrt(2.0))
    return float(S), float(z), float(p)

# =============================================================================
# 6. TREND CLASSIFICATION
# =============================================================================
def classify_trend(slope, p_value):
    if not np.isfinite(slope) or not np.isfinite(p_value):
        return NOT_SIGNIFICANT_LABEL
    if p_value < TREND_P_THRESHOLD and slope < 0:
        return SIGNIFICANT_DECREASE_LABEL
    if p_value < TREND_P_THRESHOLD and slope > 0:
        return SIGNIFICANT_INCREASE_LABEL
    return NOT_SIGNIFICANT_LABEL

# =============================================================================
# 7. CALCULATE LAKE-LEVEL TRENDS
# =============================================================================
def calculate_lake_metrics(source_df):
    print("\nCalculating lake-level Theil–Sen slopes and Mann–Kendall trends...")
    result = source_df.copy()
    years = np.arange(START_YEAR, END_YEAR + 1, dtype=float)
    slopes, intercepts, mk_s_values, mk_z_values, mk_p_values = [], [], [], [], []
    n_valid_values, trend_classes = [], []
    total = len(result)
    for row_number, (_, row) in enumerate(result.iterrows(), start=1):
        values = row[YEAR_COLUMNS].to_numpy(dtype=float)
        n_valid = int(np.isfinite(values).sum())
        if n_valid < MIN_VALID_YEARS:
            slope = intercept = mk_s = mk_z = mk_p = np.nan
        else:
            slope, intercept = calculate_sen_slope(years, values)
            mk_s, mk_z, mk_p = mann_kendall_test(values)
        slopes.append(slope)
        intercepts.append(intercept)
        mk_s_values.append(mk_s)
        mk_z_values.append(mk_z)
        mk_p_values.append(mk_p)
        n_valid_values.append(n_valid)
        trend_classes.append(classify_trend(slope, mk_p))
        if row_number % 1000 == 0 or row_number == total:
            print(f"  Processed {row_number:,}/{total:,}")
    result[VALID_YEAR_COLUMN] = n_valid_values
    result[SLOPE_COLUMN] = slopes
    result[INTERCEPT_COLUMN] = intercepts
    result[MK_S_COLUMN] = mk_s_values
    result[MK_Z_COLUMN] = mk_z_values
    result[P_COLUMN] = mk_p_values
    result[TREND_CLASS_COLUMN] = trend_classes
    return result

# =============================================================================
# 8. TREND SUMMARY
# =============================================================================
def build_trend_summary(lake_metrics):
    valid = lake_metrics[SLOPE_COLUMN].notna() & lake_metrics[P_COLUMN].notna()
    d = lake_metrics.loc[valid].copy()
    total = len(d)
    decrease = int((d[TREND_CLASS_COLUMN] == SIGNIFICANT_DECREASE_LABEL).sum())
    increase = int((d[TREND_CLASS_COLUMN] == SIGNIFICANT_INCREASE_LABEL).sum())
    nonsig = int((d[TREND_CLASS_COLUMN] == NOT_SIGNIFICANT_LABEL).sum())
    rows = [
        {"trend_class": SIGNIFICANT_DECREASE_LABEL, "count": decrease, "percentage": decrease / total * 100 if total else np.nan},
        {"trend_class": SIGNIFICANT_INCREASE_LABEL, "count": increase, "percentage": increase / total * 100 if total else np.nan},
        {"trend_class": NOT_SIGNIFICANT_LABEL, "count": nonsig, "percentage": nonsig / total * 100 if total else np.nan},
        {"trend_class": "All valid trends", "count": total, "percentage": 100.0 if total else np.nan}
    ]
    return pd.DataFrame(rows)

# =============================================================================
# 9. EXPORT
# =============================================================================
def export_results(lake_metrics, trend_summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        lake_metrics.to_excel(writer, sheet_name="Lake_metrics", index=False)
        trend_summary.to_excel(writer, sheet_name="Trend_summary", index=False)
    print(f"\nResults exported:\n{OUTPUT_XLSX}")

# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    print("=" * 72)
    print("GLOBAL LAKE TSI TREND ANALYSIS")
    print("=" * 72)
    source_df = read_data()
    print(f"Input lakes: {len(source_df):,}")
    lake_metrics = calculate_lake_metrics(source_df)
    trend_summary = build_trend_summary(lake_metrics)
    print(f"\nTemporal trend summary (MK p < {TREND_P_THRESHOLD}):")
    print(trend_summary.to_string(index=False, formatters={"percentage": lambda x: f"{x:.2f}"}))
    export_results(lake_metrics, trend_summary)
    valid = lake_metrics[SLOPE_COLUMN].notna() & lake_metrics[P_COLUMN].notna()
    sig_dec = int((lake_metrics.loc[valid, TREND_CLASS_COLUMN] == SIGNIFICANT_DECREASE_LABEL).sum())
    sig_inc = int((lake_metrics.loc[valid, TREND_CLASS_COLUMN] == SIGNIFICANT_INCREASE_LABEL).sum())
    print(f"\nValid trends: {int(valid.sum()):,}")
    print(f"Significant decreases: {sig_dec:,}")
    print(f"Significant increases: {sig_inc:,}")

if __name__ == "__main__":
    main()
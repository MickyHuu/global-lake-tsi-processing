# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:09:27 2026

@author: Min
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
# =============================================================================
# 1. PARAMETERS
# =============================================================================
INPUT_FILE = r"C:\Users\Desktop\GloTSI_zscore_results.xlsx"
SHEET_NAME = "Lake_classification"
OUTPUT_DIR = r"C:\Users\Desktop\TSI_segmented_regression"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "01_Global_TSI_zscore_segmented_regression.png")
OUTPUT_BOOT_FIG = os.path.join(OUTPUT_DIR, "02_Breakpoint_bootstrap_distribution.png")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "Global_TSI_segmented_regression_results.xlsx")
START_YEAR = 1984
END_YEAR = 2023
YEARS = np.arange(START_YEAR, END_YEAR + 1)
YEAR_COLS = [str(y) for y in YEARS]
MIN_VALID_YEARS = 2
MIN_SEGMENT_YEARS = 5
BOOTSTRAP_N = 2000
BLOCK_LENGTH = 4
RANDOM_SEED = 42
FIGSIZE = (4.2, 3.5)
DPI = 800
FONT = "Arial"
FONT_SIZE = 12
LINE_WIDTH = 2.0
CI_ALPHA = 0.18
BREAK_CI_ALPHA = 0.12
X_TICKS = np.arange(1985, 2024, 5)
COLOR_MEAN = "#222222"
COLOR_SEG = "#C94C4C"
COLOR_CI = "#777777"
COLOR_BREAK = "#C94C4C"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================================================================
# 2. READ TSI DATA
# =============================================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]
if "Hylak_id" not in df.columns:
    raise ValueError("Input file must contain Hylak_id.")
missing = [c for c in YEAR_COLS if c not in df.columns]
if missing:
    raise ValueError("Missing year columns: " + ", ".join(missing))
df[YEAR_COLS] = df[YEAR_COLS].apply(pd.to_numeric, errors="coerce")
print("=" * 78)
print("GLOBAL TSI SEGMENTED REGRESSION")
print("=" * 78)
print(f"Original lakes: {len(df):,}")
# =============================================================================
# 3. PER-LAKE Z-SCORE, 1984-2023
# =============================================================================
valid_n = df[YEAR_COLS].notna().sum(axis=1)
lake_mean = df[YEAR_COLS].mean(axis=1, skipna=True)
lake_std = df[YEAR_COLS].std(axis=1, skipna=True, ddof=0)
valid_lake = (valid_n >= MIN_VALID_YEARS) & lake_std.notna() & (lake_std > 0)
tsi = df.loc[valid_lake, YEAR_COLS].copy()
lake_mean = lake_mean.loc[valid_lake]
lake_std = lake_std.loc[valid_lake]
zscore = tsi.sub(lake_mean, axis=0).div(lake_std, axis=0)
print(f"Lakes used for z-score: {len(zscore):,}")
print(f"Lakes excluded: {len(df) - len(zscore):,}")
# =============================================================================
# 4. GLOBAL ANNUAL MEAN Z-SCORE + 95% CI
# =============================================================================
annual_mean = zscore.mean(axis=0, skipna=True).to_numpy(dtype=float)
annual_sd = zscore.std(axis=0, skipna=True, ddof=1).to_numpy(dtype=float)
annual_n = zscore.count(axis=0).to_numpy(dtype=int)
annual_se = annual_sd / np.sqrt(annual_n)
annual_ci = 1.96 * annual_se
annual_low = annual_mean - annual_ci
annual_high = annual_mean + annual_ci
annual_stats = pd.DataFrame({
    "Year": YEARS,
    "Mean_zscore": annual_mean,
    "SD": annual_sd,
    "SE": annual_se,
    "CI95_low": annual_low,
    "CI95_high": annual_high,
    "N": annual_n
})
# =============================================================================
# 5. SEGMENTED REGRESSION FUNCTIONS
# =============================================================================
x = YEARS.astype(float)
y = annual_mean.copy()
def design_matrix(x, breakpoint):
    x_center = x - START_YEAR
    hinge = np.maximum(0.0, x - breakpoint)
    return np.column_stack([
        np.ones(len(x)),
        x_center,
        hinge
    ])
def rss_for_breakpoint(breakpoint, x, y):
    X = design_matrix(x, breakpoint)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    return np.sum((y - fitted) ** 2)
def fit_segmented(x, y):
    lower = START_YEAR + MIN_SEGMENT_YEARS
    upper = END_YEAR - MIN_SEGMENT_YEARS
    grid = np.arange(lower, upper + 0.001, 0.10)
    rss_grid = np.array([
        rss_for_breakpoint(bp, x, y)
        for bp in grid
    ])
    initial_bp = grid[np.argmin(rss_grid)]
    local_lower = max(lower, initial_bp - 0.5)
    local_upper = min(upper, initial_bp + 0.5)
    result = minimize_scalar(
        lambda bp: rss_for_breakpoint(bp, x, y),
        bounds=(local_lower, local_upper),
        method="bounded",
        options={"xatol": 1e-8}
    )
    bp = float(result.x)
    X = design_matrix(x, bp)
    model = sm.OLS(y, X).fit()
    fitted = model.predict(X)
    residuals = y - fitted
    return bp, model, fitted, residuals
# =============================================================================
# 6. FIT ONE-BREAK SEGMENTED MODEL
# =============================================================================
breakpoint, seg_model, seg_fitted, residuals = fit_segmented(x, y)
beta0 = seg_model.params[0]
slope_before = seg_model.params[1]
slope_change = seg_model.params[2]
slope_after = slope_before + slope_change
se_before = seg_model.bse[1]
p_before = seg_model.pvalues[1]
cov = seg_model.cov_params()
var_after = cov[1, 1] + cov[2, 2] + 2 * cov[1, 2]
se_after = np.sqrt(var_after)
t_after = slope_after / se_after
p_after = 2 * t_dist.sf(np.abs(t_after), df=seg_model.df_resid)
se_change = seg_model.bse[2]
p_change = seg_model.pvalues[2]
# =============================================================================
# 7. SIMPLE LINEAR MODEL FOR COMPARISON
# =============================================================================
X_linear = sm.add_constant(x - START_YEAR)
linear_model = sm.OLS(y, X_linear).fit()
linear_fitted = linear_model.predict(X_linear)
linear_slope = linear_model.params[1]
linear_p = linear_model.pvalues[1]
rss_linear = np.sum((y - linear_fitted) ** 2)
rss_segmented = np.sum((y - seg_fitted) ** 2)
n_obs = len(y)
k_linear = 2
k_segmented = 4
aic_linear = n_obs * np.log(rss_linear / n_obs) + 2 * k_linear
aic_segmented = n_obs * np.log(rss_segmented / n_obs) + 2 * k_segmented
bic_linear = n_obs * np.log(rss_linear / n_obs) + k_linear * np.log(n_obs)
bic_segmented = n_obs * np.log(rss_segmented / n_obs) + k_segmented * np.log(n_obs)
r2_linear = linear_model.rsquared
r2_segmented = seg_model.rsquared
delta_aic = aic_linear - aic_segmented
delta_bic = bic_linear - bic_segmented
# =============================================================================
# 8. MOVING-BLOCK RESIDUAL BOOTSTRAP FOR BREAKPOINT CI
# =============================================================================
def moving_block_resample(values, block_length, rng):
    values = np.asarray(values)
    n = len(values)
    result = []
    while len(result) < n:
        start = rng.integers(0, n)
        block = [values[(start + j) % n] for j in range(block_length)]
        result.extend(block)
    return np.asarray(result[:n])
rng = np.random.default_rng(RANDOM_SEED)
centered_residuals = residuals - np.mean(residuals)
bootstrap_bp = []
bootstrap_pre = []
bootstrap_post = []
print("\nRunning breakpoint bootstrap...")
for b in range(BOOTSTRAP_N):
    boot_resid = moving_block_resample(
        centered_residuals,
        BLOCK_LENGTH,
        rng
    )
    y_boot = seg_fitted + boot_resid
    try:
        bp_b, model_b, _, _ = fit_segmented(x, y_boot)
        pre_b = model_b.params[1]
        post_b = model_b.params[1] + model_b.params[2]
        bootstrap_bp.append(bp_b)
        bootstrap_pre.append(pre_b)
        bootstrap_post.append(post_b)
    except Exception:
        continue
bootstrap_bp = np.asarray(bootstrap_bp)
bootstrap_pre = np.asarray(bootstrap_pre)
bootstrap_post = np.asarray(bootstrap_post)
bp_ci_low, bp_ci_high = np.percentile(bootstrap_bp, [2.5, 97.5])
pre_ci_low, pre_ci_high = np.percentile(bootstrap_pre, [2.5, 97.5])
post_ci_low, post_ci_high = np.percentile(bootstrap_post, [2.5, 97.5])
# =============================================================================
# 9. TREND REVERSAL DIAGNOSIS
# =============================================================================
is_reversal = (
    slope_before < 0 and
    slope_after > 0
)
is_significant_reversal = (
    slope_before < 0 and
    slope_after > 0 and
    p_change < 0.05
)
# =============================================================================
# 10. PRINT RESULTS
# =============================================================================
print("\n" + "=" * 78)
print("SEGMENTED REGRESSION RESULTS")
print("=" * 78)
print(f"Breakpoint: {breakpoint:.2f}")
print(f"Breakpoint 95% CI: {bp_ci_low:.2f} - {bp_ci_high:.2f}")
print(f"Approximate breakpoint year: {int(round(breakpoint))}")
print(f"Pre-break slope: {slope_before:.5f} z-score yr^-1")
print(f"Pre-break SE: {se_before:.5f}")
print(f"Pre-break p: {p_before:.6g}")
print(f"Pre-break bootstrap 95% CI: {pre_ci_low:.5f} - {pre_ci_high:.5f}")
print(f"Post-break slope: {slope_after:.5f} z-score yr^-1")
print(f"Post-break SE: {se_after:.5f}")
print(f"Post-break p: {p_after:.6g}")
print(f"Post-break bootstrap 95% CI: {post_ci_low:.5f} - {post_ci_high:.5f}")
print(f"Slope change: {slope_change:.5f} z-score yr^-1")
print(f"Slope-change SE: {se_change:.5f}")
print(f"Slope-change p: {p_change:.6g}")
print(f"\nLinear slope: {linear_slope:.5f} z-score yr^-1")
print(f"Linear trend p: {linear_p:.6g}")
print(f"Linear R2: {r2_linear:.4f}")
print(f"Segmented R2: {r2_segmented:.4f}")
print(f"Linear AIC: {aic_linear:.3f}")
print(f"Segmented AIC: {aic_segmented:.3f}")
print(f"Delta AIC (linear - segmented): {delta_aic:.3f}")
print(f"Linear BIC: {bic_linear:.3f}")
print(f"Segmented BIC: {bic_segmented:.3f}")
print(f"Delta BIC (linear - segmented): {delta_bic:.3f}")
print(f"\nDecline-to-increase reversal: {is_reversal}")
print(f"Significant slope reversal: {is_significant_reversal}")
print(f"Successful bootstrap iterations: {len(bootstrap_bp):,}/{BOOTSTRAP_N:,}")
# =============================================================================
# 11. MAIN FIGURE
# =============================================================================
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.fill_between(
    YEARS,
    annual_low,
    annual_high,
    color=COLOR_CI,
    alpha=CI_ALPHA,
    linewidth=0,
    label="95% CI",
    zorder=1
)
ax.axvspan(
    bp_ci_low,
    bp_ci_high,
    color=COLOR_BREAK,
    alpha=BREAK_CI_ALPHA,
    linewidth=0,
    label="Breakpoint 95% CI",
    zorder=0
)
ax.plot(
    YEARS,
    annual_mean,
    color=COLOR_MEAN,
    linewidth=1.6,
    label="Mean z-score",
    zorder=3
)
dense_x = np.linspace(START_YEAR, END_YEAR, 500)
dense_X = design_matrix(dense_x, breakpoint)
dense_y = dense_X @ seg_model.params
ax.plot(
    dense_x,
    dense_y,
    color=COLOR_SEG,
    linewidth=LINE_WIDTH,
    label="Segmented fit",
    zorder=4
)
ax.axvline(
    breakpoint,
    color=COLOR_BREAK,
    linestyle="--",
    linewidth=1.2,
    zorder=2
)
ax.axhline(
    0,
    color="0.45",
    linestyle=":",
    linewidth=0.9,
    zorder=0
)
ymin, ymax = ax.get_ylim()
label_y = ymax - 0.06 * (ymax - ymin)
ax.text(
    breakpoint,
    label_y,
    f"{breakpoint:.1f}",
    ha="center",
    va="top",
    fontsize=FONT_SIZE
)
ax.set_xlim(1983.5, 2023.5)
ax.set_xticks(X_TICKS)
ax.set_xlabel("Year")
ax.set_ylabel("Mean z-score")
ax.legend(frameon=False, loc="best")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(direction="out", length=4, width=0.8)
fig.tight_layout()
fig.savefig(
    OUTPUT_FIG,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)
# =============================================================================
# 12. BOOTSTRAP BREAKPOINT DISTRIBUTION
# =============================================================================
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.hist(
    bootstrap_bp,
    bins=np.arange(
        np.floor(bootstrap_bp.min()) - 0.5,
        np.ceil(bootstrap_bp.max()) + 1.0,
        1.0
    ),
    edgecolor="white",
    linewidth=0.5
)
ax.axvline(
    breakpoint,
    color=COLOR_BREAK,
    linewidth=LINE_WIDTH,
    label=f"Breakpoint = {breakpoint:.1f}"
)
ax.axvline(
    bp_ci_low,
    color="0.35",
    linestyle="--",
    linewidth=1.0
)
ax.axvline(
    bp_ci_high,
    color="0.35",
    linestyle="--",
    linewidth=1.0,
    label=f"95% CI = {bp_ci_low:.1f}-{bp_ci_high:.1f}"
)
ax.set_xlabel("Breakpoint year")
ax.set_ylabel("Bootstrap frequency")
ax.legend(frameon=False, loc="best")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(direction="out", length=4, width=0.8)
fig.tight_layout()
fig.savefig(
    OUTPUT_BOOT_FIG,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)
# =============================================================================
# 13. BREAKPOINT PROFILE
# =============================================================================
profile_years = np.arange(
    START_YEAR + MIN_SEGMENT_YEARS,
    END_YEAR - MIN_SEGMENT_YEARS + 0.001,
    0.10
)
profile_rss = np.array([
    rss_for_breakpoint(bp, x, y)
    for bp in profile_years
])
profile = pd.DataFrame({
    "Breakpoint_candidate": profile_years,
    "RSS": profile_rss
})
# =============================================================================
# 14. MODEL SUMMARY TABLE
# =============================================================================
summary = pd.DataFrame({
    "Parameter": [
        "Breakpoint",
        "Breakpoint_95CI_low",
        "Breakpoint_95CI_high",
        "Pre_break_slope",
        "Pre_break_SE",
        "Pre_break_p",
        "Pre_break_bootstrap_CI_low",
        "Pre_break_bootstrap_CI_high",
        "Post_break_slope",
        "Post_break_SE",
        "Post_break_p",
        "Post_break_bootstrap_CI_low",
        "Post_break_bootstrap_CI_high",
        "Slope_change",
        "Slope_change_SE",
        "Slope_change_p",
        "Linear_slope",
        "Linear_p",
        "Linear_R2",
        "Segmented_R2",
        "Linear_AIC",
        "Segmented_AIC",
        "Delta_AIC",
        "Linear_BIC",
        "Segmented_BIC",
        "Delta_BIC",
        "Decline_to_increase_reversal",
        "Significant_slope_reversal",
        "Bootstrap_successful_N"
    ],
    "Value": [
        breakpoint,
        bp_ci_low,
        bp_ci_high,
        slope_before,
        se_before,
        p_before,
        pre_ci_low,
        pre_ci_high,
        slope_after,
        se_after,
        p_after,
        post_ci_low,
        post_ci_high,
        slope_change,
        se_change,
        p_change,
        linear_slope,
        linear_p,
        r2_linear,
        r2_segmented,
        aic_linear,
        aic_segmented,
        delta_aic,
        bic_linear,
        bic_segmented,
        delta_bic,
        is_reversal,
        is_significant_reversal,
        len(bootstrap_bp)
    ]
})
bootstrap_results = pd.DataFrame({
    "Bootstrap": np.arange(1, len(bootstrap_bp) + 1),
    "Breakpoint": bootstrap_bp,
    "Pre_break_slope": bootstrap_pre,
    "Post_break_slope": bootstrap_post
})
# =============================================================================
# 15. EXPORT EXCEL
# =============================================================================
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    annual_stats.to_excel(
        writer,
        sheet_name="Annual_zscore",
        index=False
    )
    summary.to_excel(
        writer,
        sheet_name="Segmented_summary",
        index=False
    )
    bootstrap_results.to_excel(
        writer,
        sheet_name="Breakpoint_bootstrap",
        index=False
    )
    profile.to_excel(
        writer,
        sheet_name="Breakpoint_profile",
        index=False
    )
print("\n" + "=" * 78)
print("OUTPUT")
print("=" * 78)
print(f"Main figure:\n{OUTPUT_FIG}")
print(f"\nBootstrap figure:\n{OUTPUT_BOOT_FIG}")
print(f"\nExcel results:\n{OUTPUT_EXCEL}")
plt.show()
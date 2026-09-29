# -*- coding: utf-8 -*-
"""
Continental TSI breakpoint

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# =============================================================================
# 1. PATHS
# =============================================================================
BASE_DIR = (r"F:\TSI_data")
RESULT_FILE = os.path.join(BASE_DIR,"Lake_continent_breakpoint_results.xlsx")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Lake_breakpoint_continent_attribution",
    "01_Breakpoint_figures"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FIGURE = os.path.join(OUTPUT_DIR, "Continental_TSI_breakpoints.png")
SLOPE_OUTPUT = os.path.join(OUTPUT_DIR, "Continental_TSI_segmented_slopes.xlsx")

# =============================================================================
# 2. PARAMETERS
# =============================================================================
START_YEAR, END_YEAR = 1984, 2023
DPI = 800
FIGSIZE = (15, 6)

FONT = "Arial"
FONT_SIZE = 16
LEGEND_FONT_SIZE = 13

PLOT_CONTINENTS = [
    "North America", "Europe", "Asia",
    "Africa", "South America", "Oceania"
]

MEAN_COLOR = "black"
CI_COLOR = "#BFBFBF"
FIT_COLOR = "#C94C4C"
BREAK_COLOR = "#555555"

MEAN_LINE_WIDTH = 1.6
FIT_LINE_WIDTH = 2.0
BREAK_LINE_WIDTH = 1.0
CI_ALPHA = 0.35

# Slope annotation
SHOW_SLOPE_ANNOTATION = True
SLOPE_TEXT_X = 0.035
SLOPE_TEXT_Y = 0.18
SLOPE_TEXT_FONT_SIZE = 13
SLOPE_TEXT_LINE_SPACING = 1.0
SLOPE_DECIMALS = 3
SHOW_POSITIVE_SIGN = True

# Layout
LEFT, RIGHT = 0.075, 0.985
BOTTOM, TOP = 0.085, 0.955
WSPACE, HSPACE = 0.18, 0.22

plt.rcParams.update({
    "font.family": FONT,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# =============================================================================
# 3. READ DATA
# =============================================================================
if not os.path.exists(RESULT_FILE):
    raise FileNotFoundError(f"Not found：\n{RESULT_FILE}")

annual_tsi = pd.read_excel(RESULT_FILE, sheet_name="Annual_TSI")
breakpoint_summary = pd.read_excel(RESULT_FILE, sheet_name="Breakpoint_summary")

required_annual = {"Scope", "Year", "Mean_TSI", "CI95_low", "CI95_high"}
required_break = {"Scope", "Break_year_used"}

missing_annual = required_annual - set(annual_tsi.columns)
missing_break = required_break - set(breakpoint_summary.columns)

if missing_annual:
    raise KeyError(f"Annual_TSI missing：{sorted(missing_annual)}")
if missing_break:
    raise KeyError(f"Breakpoint_summary missing：{sorted(missing_break)}")

# =============================================================================
# 4. SEGMENTED REGRESSION
# =============================================================================
def hinge_design(x, breakpoint):
    x = np.asarray(x, dtype=float)
    return np.column_stack([
        np.ones(len(x)),
        x - START_YEAR,
        np.maximum(0.0, x - breakpoint)
    ])


def fit_segmented_line(x, y, breakpoint):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)

    X = hinge_design(x[valid], breakpoint)
    beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]

    fitted = hinge_design(x, breakpoint) @ beta
    pre_slope = beta[1]
    post_slope = beta[1] + beta[2]

    return fitted, pre_slope, post_slope


def format_slope(value):
    if SHOW_POSITIVE_SIGN:
        return f"{value:+.{SLOPE_DECIMALS}f}"
    return f"{value:.{SLOPE_DECIMALS}f}"

# =============================================================================
# 5. DRAW 2 × 3 FIGURE
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True)
axes = axes.ravel()

slope_records = []

for ax, continent in zip(axes, PLOT_CONTINENTS):

    d = annual_tsi[annual_tsi["Scope"] == continent].copy()
    bp_row = breakpoint_summary[breakpoint_summary["Scope"] == continent]

    if d.empty or bp_row.empty:
        ax.set_visible(False)
        continue

    d = d.sort_values("Year")

    x = d["Year"].to_numpy(float)
    y = d["Mean_TSI"].to_numpy(float)
    low = d["CI95_low"].to_numpy(float)
    high = d["CI95_high"].to_numpy(float)
    bp = int(bp_row["Break_year_used"].iloc[0])

    fitted, pre_slope, post_slope = fit_segmented_line(x, y, bp)

    slope_records.append({
        "Continent": continent,
        "Breakpoint": bp,
        "Pre_start": START_YEAR,
        "Pre_end": bp,
        "Pre_slope": pre_slope,
        "Post_start": bp,
        "Post_end": END_YEAR,
        "Post_slope": post_slope
    })

    ax.fill_between(
        x, low, high,
        color=CI_COLOR,
        alpha=CI_ALPHA,
        linewidth=0,
        label="95% CI",
        zorder=1
    )

    ax.plot(
        x, y,
        color=MEAN_COLOR,
        linewidth=MEAN_LINE_WIDTH,
        label="Mean TSI",
        zorder=2
    )

    ax.plot(
        x, fitted,
        color=FIT_COLOR,
        linewidth=FIT_LINE_WIDTH,
        label="Segmented fit",
        zorder=3
    )

    ax.axvline(
        bp,
        color=BREAK_COLOR,
        linestyle="--",
        linewidth=BREAK_LINE_WIDTH,
        label="Break year",
        zorder=4
    )

    if SHOW_SLOPE_ANNOTATION:
        slope_text = (
            f"{START_YEAR}–{bp}: {format_slope(pre_slope)} yr$^{{-1}}$\n"
            f"{bp}–{END_YEAR}: {format_slope(post_slope)} yr$^{{-1}}$"
        )

        ax.text(
            SLOPE_TEXT_X, SLOPE_TEXT_Y, slope_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=SLOPE_TEXT_FONT_SIZE,
            linespacing=SLOPE_TEXT_LINE_SPACING,
            color=FIT_COLOR,
            zorder=5
        )

    ax.set_xlim(1983.5, 2023.5)
    ax.set_xticks(np.arange(1985, 2024, 10))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_title(f"{continent} (break = {bp})")
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        direction="out",
        length=4,
        width=0.8
    )

    if continent == "Asia":
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0, 2]

        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="lower right",
            bbox_to_anchor=(1.0, 0.60),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=2.0,
            handletextpad=0.5,
            labelspacing=0.35
        )

# =============================================================================
# 6. AXIS LABELS + LAYOUT
# =============================================================================
for ax in axes[3:]:
    ax.set_xlabel("Year")

for ax in axes[[0, 3]]:
    ax.set_ylabel("Mean TSI")

plt.subplots_adjust(
    left=LEFT,
    right=RIGHT,
    bottom=BOTTOM,
    top=TOP,
    wspace=WSPACE,
    hspace=HSPACE
)

# =============================================================================
# 7. SAVE
# =============================================================================
fig.savefig(
    OUTPUT_FIGURE,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)

OUTPUT_PDF = os.path.splitext(OUTPUT_FIGURE)[0] + ".pdf"

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor="white"
)

# =============================================================================
# 8. EXPORT SLOPES
# =============================================================================
slope_df = pd.DataFrame(slope_records)
slope_df.to_excel(SLOPE_OUTPUT, index=False)

print("\n" + "=" * 72)
print("CONTINENTAL SEGMENTED TSI SLOPES")
print("=" * 72)

print(
    slope_df[
        ["Continent", "Breakpoint", "Pre_slope", "Post_slope"]
    ].round(4).to_string(index=False)
)

print(f"\nFigure saved:\n{OUTPUT_FIGURE}")
print(f"\nPDF saved:\n{OUTPUT_PDF}")
print(f"\nSlope statistics saved:\n{SLOPE_OUTPUT}")

plt.show()
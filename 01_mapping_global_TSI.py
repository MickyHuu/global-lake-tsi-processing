# -*- coding: utf-8 -*-
"""
Plot global long-term mean TSI.

"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker


# =============================================================================
# 1. USER-CUSTOMIZABLE PARAMETERS
# =============================================================================

INPUT_XLS = Path(r"F:\01_lakeTSI_40y.xls")
TARGET_SHEET = "sheet1"
WORLD_SHP = Path(r"F:\World_countries.shp")
OUTPUT_DIR = Path(r"C:\TSI_outputs")
OUTPUT_FIGURE_NAME = ("Figure1a_longterm_mean_TSI.png")

EXPORT_DPI = 800
SHOW_FIGURE = True
ID_COLUMN = "Hylak_id"
LON_COLUMN = "Pour_long"
LAT_COLUMN = "Pour_lat"
AVG_TSI_COLUMN = "avgTSI"

# =============================================================================
# 2. CANVAS AND AXES
# =============================================================================

FIG_WIDTH = 16
FIG_HEIGHT = 8
FIGURE_FACE_COLOR = "white"
AX_LEFT = 0.055
AX_BOTTOM = 0.15
AX_WIDTH = 0.89
AX_HEIGHT = 0.76

# =============================================================================
# 3. FONT
# =============================================================================

FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 16
COLORBAR_LABEL_FONT_SIZE = 16
COLORBAR_TICK_FONT_SIZE = 16

# =============================================================================
# 4. MAP EXTENT AND GRATICULES
# =============================================================================

LON_MIN, LON_MAX = -180, 180
LAT_MIN, LAT_MAX = -60, 85

LON_INTERVAL = 60
LAT_INTERVAL = 30

SHOW_LONGITUDE_LABELS = True
SHOW_LATITUDE_LABELS = True

SHOW_DATELINE_LONGITUDE_LABELS = False

LONGITUDE_LABEL_PAD = 6
LATITUDE_LABEL_PAD = 8

LATITUDE_LABEL_ROTATION = 90
LATITUDE_LABEL_H_ALIGN = "center"
LATITUDE_LABEL_V_ALIGN = "center"

GRID_LINE_COLOR = "white"
GRID_LINE_WIDTH = 0.65
GRID_LINE_STYLE = "--"
GRID_LINE_ALPHA = 0.75

# =============================================================================
# 5. COUNTRY BACKGROUND
# =============================================================================

COUNTRY_FACE_COLOR = "#E8E8E8"
COUNTRY_EDGE_COLOR = "#FFFFFF"
COUNTRY_EDGE_WIDTH = 0.45

# =============================================================================
# 6. LAKE POINTS
# =============================================================================

POINT_SIZE = 10
POINT_ALPHA = 0.88
POINT_EDGE_COLOR = "none"
POINT_EDGE_WIDTH = 0

SORT_HIGH_VALUES_ON_TOP = True

# =============================================================================
# 7. COLORMAP AND COLORBAR
# =============================================================================

TSI_CMAP_NAME = "YlGnBu_r"

TSI_VMIN = 0
TSI_VMAX = 70

TSI_TICKS = np.arange(
    0,
    71,
    10
)

COLORBAR_ORIENTATION = "horizontal"
COLORBAR_LEFT = 0.30
COLORBAR_BOTTOM = 0.075
COLORBAR_WIDTH = 0.40
COLORBAR_HEIGHT = 0.025
COLORBAR_LABEL = "Long-term mean TSI"
COLORBAR_EXTEND = "both"

# =============================================================================
# 8. TITLE
# =============================================================================

PANEL_LABEL = ""
TITLE = ""

TITLE_PAD = 12
TITLE_WEIGHT = "bold"

TITLE_X = -0.01

# =============================================================================
# 9. GENERAL STYLE
# =============================================================================

def set_style() -> None:

    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": TICK_FONT_SIZE,

        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,

        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "savefig.facecolor": FIGURE_FACE_COLOR,
        "figure.facecolor": FIGURE_FACE_COLOR
    })

# =============================================================================
# 10. CHECK INPUT FILES
# =============================================================================

def check_input_files() -> None:

    if not INPUT_XLS.exists():

        raise FileNotFoundError(
            f"Input workbook not found:\n"
            f"{INPUT_XLS}"
        )

    if not WORLD_SHP.exists():

        raise FileNotFoundError(
            f"World shapefile not found:\n"
            f"{WORLD_SHP}\n\n"
            "Please make sure .shp, .shx, .dbf "
            "and .prj files are all present."
        )

# =============================================================================
# 11. READ LAKE DATA
# =============================================================================

def read_original_lake_data() -> pd.DataFrame:

    try:

        df = pd.read_excel(
            INPUT_XLS,
            sheet_name=TARGET_SHEET,
            engine="xlrd"
        )

    except ImportError as exc:

        raise ImportError(
            "Reading the .xls workbook requires xlrd.\n"
            "Install with:\n"
            "pip install xlrd"
        ) from exc


    required = {
        ID_COLUMN,
        LON_COLUMN,
        LAT_COLUMN,
        AVG_TSI_COLUMN
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:

        raise KeyError(
            "The target sheet is missing "
            "required columns:\n"
            + ", ".join(missing)
        )

    for column in [
        LON_COLUMN,
        LAT_COLUMN,
        AVG_TSI_COLUMN
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    print("=" * 75)
    print("ORIGINAL LONG-TERM MEAN TSI")
    print("=" * 75)

    print(
        f"Total lake records: "
        f"{len(df):,}"
    )

    print(
        f"Valid TSI: "
        f"{df[AVG_TSI_COLUMN].notna().sum():,}"
    )

    print(
        f"TSI range: "
        f"{df[AVG_TSI_COLUMN].min():.2f} – "
        f"{df[AVG_TSI_COLUMN].max():.2f}"
    )

    print(
        f"Mean TSI: "
        f"{df[AVG_TSI_COLUMN].mean():.2f}"
    )

    return df

# =============================================================================
# 12. READ WORLD BOUNDARY
# =============================================================================

def read_world_boundary() -> gpd.GeoDataFrame:

    world = gpd.read_file(WORLD_SHP)
    world = world.loc[world.geometry.notna()&~world.geometry.is_empty].copy()

    if world.crs is None:

        warnings.warn(
            "The world shapefile has no CRS. "
            "EPSG:4326 is assigned."
        )

        world = world.set_crs(
            "EPSG:4326"
        )

    else:

        world = world.to_crs(
            "EPSG:4326"
        )

    name_candidates = [
        "NAME",
        "NAME_EN",
        "ADMIN",
        "COUNTRY",
        "SOVEREIGNT",
        "NAME_LONG",
        "CNTRY_NAME"
    ]

    upper_lookup = {
        str(column).upper():
            column
        for column in world.columns
    }

    name_field = next(
        (
            upper_lookup[
                name.upper()
            ]

            for name in name_candidates

            if name.upper()
            in upper_lookup
        ),
        None
    )


    if name_field is not None:

        names = (
            world[name_field]
            .astype(str)
            .str.lower()
        )

        world = world.loc[
            ~names.str.contains(
                "antarctica",
                regex=True,
                na=False
            )
        ].copy()


    return world

# =============================================================================
# 13. ADD GRATICULES
# =============================================================================

def add_graticules(ax) -> None:

    x_grid_ticks = np.arange(
        LON_MIN,
        LON_MAX + 0.1,
        LON_INTERVAL
    )

    y_ticks = np.arange(
        LAT_MIN,
        LAT_MAX + 0.1,
        LAT_INTERVAL
    )

    if SHOW_DATELINE_LONGITUDE_LABELS:

        x_label_ticks = (
            x_grid_ticks
        )

    else:

        x_label_ticks = (
            x_grid_ticks[
                (x_grid_ticks > LON_MIN)
                &
                (x_grid_ticks < LON_MAX)
            ]
        )


    ax.set_xticks(
        x_label_ticks,
        crs=ccrs.PlateCarree()
    )

    ax.set_yticks(
        y_ticks,
        crs=ccrs.PlateCarree()
    )


    ax.xaxis.set_major_formatter(
        cticker.LongitudeFormatter()
    )

    ax.yaxis.set_major_formatter(
        cticker.LatitudeFormatter()
    )

    ax.tick_params(
        axis="x",
        labelsize=TICK_FONT_SIZE,
        length=4,
        width=0.8,
        pad=LONGITUDE_LABEL_PAD
    )

    ax.tick_params(
        axis="y",
        labelsize=TICK_FONT_SIZE,
        length=4,
        width=0.8,
        pad=LATITUDE_LABEL_PAD
    )

    plt.setp(
        ax.get_yticklabels(),
        rotation=LATITUDE_LABEL_ROTATION,
        ha=LATITUDE_LABEL_H_ALIGN,
        va=LATITUDE_LABEL_V_ALIGN,
        rotation_mode="anchor"
    )

    if not SHOW_LONGITUDE_LABELS:

        ax.tick_params(
            axis="x",
            labelbottom=False
        )

    if not SHOW_LATITUDE_LABELS:

        ax.tick_params(
            axis="y",
            labelleft=False
        )

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        xlocs=x_grid_ticks,
        ylocs=y_ticks,
        draw_labels=False,
        linewidth=GRID_LINE_WIDTH,
        color=GRID_LINE_COLOR,
        alpha=GRID_LINE_ALPHA,
        linestyle=GRID_LINE_STYLE,
        zorder=0
    )


# =============================================================================
# 14. PLOT LONG-TERM MEAN TSI
# =============================================================================

def plot_original_map(
    lake_df: pd.DataFrame,
    world: gpd.GeoDataFrame
) -> Path:

    set_style()


    valid_map = (
        lake_df[LON_COLUMN].between(
            -180,
            180
        )
        &
        lake_df[LAT_COLUMN].between(
            -90,
            90
        )
        &
        lake_df[
            AVG_TSI_COLUMN
        ].notna()
    )


    map_df = lake_df.loc[
        valid_map
    ].copy()


    print(
        f"Valid lakes plotted: "
        f"{len(map_df):,}"
    )

    if SORT_HIGH_VALUES_ON_TOP:

        map_df = (
            map_df.sort_values(
                AVG_TSI_COLUMN,
                ascending=True
            )
        )


    cmap = plt.get_cmap(TSI_CMAP_NAME)


    norm = Normalize(vmin=TSI_VMIN,vmax=TSI_VMAX)

    fig = plt.figure(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        ),
        facecolor=
            FIGURE_FACE_COLOR
    )

    ax = fig.add_axes(
        [
            AX_LEFT,
            AX_BOTTOM,
            AX_WIDTH,
            AX_HEIGHT
        ],
        projection=
            ccrs.PlateCarree()
    )


    ax.set_extent(
        [
            LON_MIN,
            LON_MAX,
            LAT_MIN,
            LAT_MAX
        ],
        crs=
            ccrs.PlateCarree()
    )

    ax.add_geometries(
        world.geometry,
        crs=ccrs.PlateCarree(),
        facecolor=COUNTRY_FACE_COLOR,
        edgecolor=COUNTRY_EDGE_COLOR,
        linewidth=COUNTRY_EDGE_WIDTH,
        zorder=1
    )


    add_graticules(ax)

    scatter = ax.scatter(
        map_df[LON_COLUMN],
        map_df[LAT_COLUMN],
        c=map_df[AVG_TSI_COLUMN],
        cmap=cmap,
        norm=norm,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_EDGE_WIDTH,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=3
    )


    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    ax.set_title(
        f"{PANEL_LABEL}  {TITLE}",
        loc="left",
        x=TITLE_X,
        fontsize=TITLE_FONT_SIZE,
        fontweight=TITLE_WEIGHT,
        pad=TITLE_PAD
    )

    cax = fig.add_axes(
        [
            COLORBAR_LEFT,
            COLORBAR_BOTTOM,
            COLORBAR_WIDTH,
            COLORBAR_HEIGHT
        ]
    )


    colorbar = fig.colorbar(
        scatter,
        cax=cax,
        orientation=COLORBAR_ORIENTATION,
        extend=COLORBAR_EXTEND
    )


    colorbar.set_label(
        COLORBAR_LABEL,
        fontsize=COLORBAR_LABEL_FONT_SIZE,
        labelpad=5
    )

    colorbar.set_ticks(
        TSI_TICKS
    )

    colorbar.ax.tick_params(
        labelsize=COLORBAR_TICK_FONT_SIZE,
        length=4,
        width=0.8
    )

    colorbar.outline.set_linewidth(0.7)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        OUTPUT_DIR
        /
        OUTPUT_FIGURE_NAME
    )


    fig.savefig(
        output_path,

        dpi=
            EXPORT_DPI,

        bbox_inches=
            "tight",

        facecolor=
            FIGURE_FACE_COLOR
    )


    print(
        f"PNG figure exported:\n"
        f"{output_path}"
    )


    if SHOW_FIGURE:

        plt.show()

    else:

        plt.close(fig)


    return output_path


# =============================================================================
# 15. MAIN
# =============================================================================

def main() -> None:

    check_input_files()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    lake_df = (read_original_lake_data())
    world = (read_world_boundary())
    plot_original_map(lake_df=lake_df,world=world)

    print(
        f"Map variable: "
        f"{AVG_TSI_COLUMN}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


# =============================================================================
# 16. RUN
# =============================================================================

if __name__ == "__main__":
    main()
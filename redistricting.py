"""
Population-Bisecting Splitline Redistricting v14
======================
Changes from v13:
  - Stage 8 fully redesigned: raster city-lights map replaces scatter plot
    - np.histogram2d per district → city-lights population density layer
    - Voronoi nearest-centroid grid for district boundary lines
    - Official Census TIGER state shapefile for the state outline
      (eliminates interior white-line artifacts from census block edge mismatches)
    - State raster mask cached to data/states/state_mask_{FIPS}_{W}x{H}.npy
  - Dev/district_viz.py sandbox for iterating on visualization (4-panel output)
  - Multi-state tested: Colorado, Texas, Illinois (PASS); California (FAIL — 52 districts)

Changes from v12b (v13):
  - Multi-state data layout: all downloaded inputs go to data/{FIPS}/
    (zip, extracted blocks/, centroids_cache.csv) — shared across runs,
    never re-downloaded if present
  - All generated outputs go to output/redistricting_{STATE}_{VERSION}/
  - Project root stays clean: only source files tracked by git

Requirements:
    pip install geopandas matplotlib shapely scipy numpy pandas tqdm
    pip install reportlab pypdf pyproj pillow

To run for any state, change STATE_NAME below, then:
    python redistricting.py

To regenerate maps/PDF only (reuse saved computation):
    python redistricting.py  # will auto-detect checkpoint
"""

# ── Startup import check ───────────────────────────────────────
import sys

REQUIRED = {
    "numpy":      "numpy",
    "pandas":     "pandas",
    "geopandas":  "geopandas",
    "matplotlib": "matplotlib",
    "shapely":    "shapely",
    "scipy":      "scipy",
    "reportlab":  "reportlab",
    "pypdf":      "pypdf",
    "pyproj":     "pyproj",
    "PIL":        "pillow",
}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)

if missing:
    print("\n" + "="*60)
    print("ERROR: Missing required packages.")
    print("Run this command to install everything needed:\n")
    print(f"  pip install {' '.join(missing)}\n")
    print("Then run the script again.")
    print("="*60 + "\n")
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────
import argparse
import colorsys
import logging
import math
import os
import pickle
import re
import urllib.request
import zipfile
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from shapely.wkt import loads as wkt_loads
from pyproj import Transformer
from PIL import Image as PILImage

# ── ReportLab — alias letter IMMEDIATELY, before any other code ──
from reportlab.lib.pagesizes import letter as LETTER_SIZE
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import (HexColor, black, white,
                                   HexColor as HC)
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable,
                                 PageBreak, Image as RLImage,
                                 KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from pypdf import PdfWriter, PdfReader

# ── Color constants (defined once, never overwritten) ──────────
PDF_NAVY  = HC("#1B3A5C")
PDF_BLUE  = HC("#4E79A7")
PDF_LGRAY = HC("#F8F8F5")
PDF_MGRAY = HC("#CCCCCC")
PDF_ORANGE= HC("#F28E2B")
PDF_GREEN = HC("#59A14F")
PDF_BLACK = black
PDF_WHITE = white

# ═══════════════════════════════════════════════════════════════
STATE_NAME = "California"
AUTHOR     = "Steve Stanzel, Boulder CO"
# ═══════════════════════════════════════════════════════════════

VERSION = "v15"

STATES = {
    "Alabama":        {"fips": "01", "districts": 7},
    "Alaska":         {"fips": "02", "districts": 1},
    "Arizona":        {"fips": "04", "districts": 9},
    "Arkansas":       {"fips": "05", "districts": 4},
    "California":     {"fips": "06", "districts": 52},
    "Colorado":       {"fips": "08", "districts": 8},
    "Connecticut":    {"fips": "09", "districts": 5},
    "Delaware":       {"fips": "10", "districts": 1},
    "Florida":        {"fips": "12", "districts": 28},
    "Georgia":        {"fips": "13", "districts": 14},
    "Hawaii":         {"fips": "15", "districts": 2},
    "Idaho":          {"fips": "16", "districts": 2},
    "Illinois":       {"fips": "17", "districts": 17},
    "Indiana":        {"fips": "18", "districts": 9},
    "Iowa":           {"fips": "19", "districts": 4},
    "Kansas":         {"fips": "20", "districts": 4},
    "Kentucky":       {"fips": "21", "districts": 6},
    "Louisiana":      {"fips": "22", "districts": 6},
    "Maine":          {"fips": "23", "districts": 2},
    "Maryland":       {"fips": "24", "districts": 8},
    "Massachusetts":  {"fips": "25", "districts": 9},
    "Michigan":       {"fips": "26", "districts": 13},
    "Minnesota":      {"fips": "27", "districts": 8},
    "Mississippi":    {"fips": "28", "districts": 4},
    "Missouri":       {"fips": "29", "districts": 8},
    "Montana":        {"fips": "30", "districts": 2},
    "Nebraska":       {"fips": "31", "districts": 3},
    "Nevada":         {"fips": "32", "districts": 4},
    "New Hampshire":  {"fips": "33", "districts": 2},
    "New Jersey":     {"fips": "34", "districts": 12},
    "New Mexico":     {"fips": "35", "districts": 3},
    "New York":       {"fips": "36", "districts": 26},
    "North Carolina": {"fips": "37", "districts": 14},
    "North Dakota":   {"fips": "38", "districts": 1},
    "Ohio":           {"fips": "39", "districts": 15},
    "Oklahoma":       {"fips": "40", "districts": 5},
    "Oregon":         {"fips": "41", "districts": 6},
    "Pennsylvania":   {"fips": "42", "districts": 17},
    "Rhode Island":   {"fips": "44", "districts": 2},
    "South Carolina": {"fips": "45", "districts": 7},
    "South Dakota":   {"fips": "46", "districts": 1},
    "Tennessee":      {"fips": "47", "districts": 9},
    "Texas":          {"fips": "48", "districts": 38},
    "Utah":           {"fips": "49", "districts": 4},
    "Vermont":        {"fips": "50", "districts": 1},
    "Virginia":       {"fips": "51", "districts": 11},
    "Washington":     {"fips": "53", "districts": 10},
    "West Virginia":  {"fips": "54", "districts": 2},
    "Wisconsin":      {"fips": "55", "districts": 8},
    "Wyoming":        {"fips": "56", "districts": 1},
}

APPORTIONMENT_YEAR = 2020

# ── Global constants needed before per-state setup ────────────
STATES_DIR = os.path.join("data", "states")  # shared across all states

MAP_COLORS = [
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
    "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC", "#D37295", "#A0CBE8",
    "#86BCB6", "#F1CE63", "#B6992D", "#499894",
]

TIGER_STATE_URL = "https://www2.census.gov/geo/tiger/TIGER2020/STATE/tl_2020_us_state.zip"
TIGER_STATE_ZIP = os.path.join(STATES_DIR, "tl_2020_us_state.zip")
TIGER_STATE_SHP = os.path.join(STATES_DIR, "tl_2020_us_state.shp")

# ── Argument parsing — runs before per-state pipeline setup ───
_parser = argparse.ArgumentParser(
    prog="redistricting.py",
    description="Population-bisecting splitline redistricting",
)
_parser.add_argument("--full", action="store_true",
                     help="Generate full report (process pages, CSVs, PDFs)")
_parser.add_argument("--state", default=None, metavar="STATE",
                     help="Override STATE_NAME (e.g. 'Virginia')")
_parser.add_argument("--usa_districts", action="store_true",
                     help="Run all 50 states quick mode and bundle district maps into one PDF")
_parser.add_argument("--usa_full", action="store_true",
                     help="Run full pipeline for all 50 states; each in its own output folder")
_args = _parser.parse_args()
FULL_REPORT = _args.full

if _args.state:
    if _args.state not in STATES:
        print(f"ERROR: '--state {_args.state}' not in STATES dict.")
        sys.exit(1)
    STATE_NAME = _args.state


def _run_usa_mode(districts_mode: bool, full_mode: bool) -> None:
    """Step through all 50 states alphabetically via subprocess, then bundle if districts_mode."""
    import subprocess as _sp
    _states_sorted = sorted(STATES.keys())
    _failed: list[str] = []
    _passed: list[str] = []
    _label = "USA Districts" if districts_mode else "USA Full"

    print(f"\n{'='*60}")
    print(f"{_label} — {len(_states_sorted)} states")
    print(f"{'='*60}\n")

    for _state in _states_sorted:
        _n = STATES[_state]["districts"]
        print(f"\n{'─'*60}")
        print(f"  {_state} ({_n} district{'s' if _n != 1 else ''})")
        print(f"{'─'*60}")
        _cmd = [sys.executable, __file__, "--state", _state]
        if full_mode:
            _cmd.append("--full")
        _result = _sp.run(_cmd)
        if _result.returncode == 0:
            _passed.append(_state)
        else:
            _failed.append(_state)
            print(f"\n  *** {_state} FAILED (returncode={_result.returncode}) ***")

    print(f"\n{'='*60}")
    print(f"{_label} complete: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print(f"  Failed: {', '.join(_failed)}")
    print(f"{'='*60}\n")

    if districts_mode:
        _bundle_usa_districts_pdf(_states_sorted)


def _bundle_usa_districts_pdf(states_sorted: list[str]) -> None:
    """Collect per-state district map PNGs and assemble into one alphabetical PDF."""
    from reportlab.platypus import (SimpleDocTemplate, Image as _RLImg,
                                    Spacer, Paragraph, PageBreak)
    from reportlab.lib.pagesizes import letter as _LETTER
    from reportlab.lib.units import inch as _inch
    from reportlab.lib.styles import getSampleStyleSheet as _gss
    from PIL import Image as _PIL

    _out_pdf = os.path.join("output", f"usa_districts_{VERSION}.pdf")
    os.makedirs("output", exist_ok=True)
    _doc = SimpleDocTemplate(
        _out_pdf, pagesize=_LETTER,
        leftMargin=0.4 * _inch, rightMargin=0.4 * _inch,
        topMargin=0.4 * _inch, bottomMargin=0.4 * _inch,
    )
    _styles = _gss()
    _story: list = []
    _missing: list[str] = []

    _story.append(Paragraph(
        f"U.S. Congressional District Maps — {APPORTIONMENT_YEAR} Census Apportionment",
        _styles["Title"],
    ))
    _story.append(Paragraph(
        f"Generated by: {AUTHOR}  ·  Version {VERSION}",
        _styles["Normal"],
    ))
    _story.append(Spacer(1, 0.15 * _inch))

    for _state in states_sorted:
        _slug = _state.replace(" ", "_")
        _png = os.path.join(
            "output", f"redistricting_{_slug}_{VERSION}",
            "assets", f"districts_{_slug}_{VERSION}.png",
        )
        if not os.path.exists(_png):
            _missing.append(_state)
            continue
        _pil_w, _pil_h = _PIL.open(_png).size
        _max_w = 7.7 * _inch
        _max_h = 9.5 * _inch
        _w = _max_w
        _h = _w * _pil_h / _pil_w
        if _h > _max_h:
            _h = _max_h
            _w = _h * _pil_w / _pil_h
        _story.append(_RLImg(_png, width=_w, height=_h))
        _story.append(PageBreak())

    if _missing:
        _story.append(Paragraph(
            "States not included (run failed or PNG not found):<br/>"
            + ", ".join(_missing),
            _styles["Normal"],
        ))

    _doc.build(_story)
    print(f"Saved: {_out_pdf}")
    if _missing:
        print(f"  Not included ({len(_missing)}): {', '.join(_missing)}")


if _args.usa_districts or _args.usa_full:
    _run_usa_mode(_args.usa_districts, _args.usa_full)
    sys.exit(0)

# ── Validate ──────────────────────────────────────────────────
if STATE_NAME not in STATES:
    print(f"ERROR: '{STATE_NAME}' not in STATES.")
    sys.exit(1)

STATE       = STATES[STATE_NAME]
STATE_FIPS  = STATE["fips"]
N_DISTRICTS = STATE["districts"]

_cities_path = os.path.join("data", "cities.csv")
_cities_df   = pd.read_csv(_cities_path) if os.path.exists(_cities_path) else pd.DataFrame()
if _cities_df.empty or STATE_NAME not in _cities_df["state"].values:
    print(f"  cities.csv missing or has no entries for {STATE_NAME} — running build_cities.py...")
    import subprocess
    subprocess.run([sys.executable, "build_cities.py"], check=True)
    _cities_df = pd.read_csv(_cities_path)
_state_rows  = _cities_df[_cities_df["state"] == STATE_NAME].head(N_DISTRICTS * 10)
# Include population for scoring; fall back to 1 if the column is absent (pre-rebuild CSV)
_pop_col = _state_rows["population"].astype(int) if "population" in _state_rows.columns else [1] * len(_state_rows)
STATE_CITIES = list(zip(_state_rows["city"], _state_rows["lat"], _state_rows["lon"], _pop_col))

FIRST_CUT_ANGLE = 135.0
ALT_CUT_ANGLE   = 45.0
MAX_TOTAL_DEV   = 0.01
SEARCH_RADIUS         = 45.0
SWAP_ROUNDS_PER_DIST  = 25
N_SWAP_ROUNDS         = SWAP_ROUNDS_PER_DIST * N_DISTRICTS

if N_DISTRICTS == 1:
    MAX_DEPTH = 1; MAX_SPLIT_ERROR = MAX_TOTAL_DEV
else:
    MAX_DEPTH = math.ceil(math.log2(N_DISTRICTS))
    MAX_SPLIT_ERROR = MAX_TOTAL_DEV / MAX_DEPTH

STATE_SLUG = STATE_NAME.replace(" ", "_")
DATA_DIR   = os.path.join("data", STATE_FIPS)
ZIP_PATH   = os.path.join(DATA_DIR, f"tl_2020_{STATE_FIPS}_tabblock20.zip")
SHP_DIR    = os.path.join(DATA_DIR, "blocks")
CACHE_CSV  = os.path.join(DATA_DIR, "centroids_cache.csv")
OUTPUT_DIR  = os.path.join("output", f"redistricting_{STATE_SLUG}_{VERSION}")
ASSETS_DIR  = os.path.join(OUTPUT_DIR, "assets")
LOGS_DIR    = os.path.join(OUTPUT_DIR, "logs")
DATA_OUT_DIR = os.path.join(OUTPUT_DIR, "data")
CHECKPOINT  = os.path.join(DATA_OUT_DIR, f"checkpoint_{STATE_SLUG}.npy")

# Backward-compat: fall back to the previous version's checkpoint if this one is absent
PREV_VERSION     = f"v{int(VERSION[1:])-1}"
_PREV_OUT_DIR    = os.path.join("output", f"redistricting_{STATE_SLUG}_{PREV_VERSION}")
_PREV_CHECKPOINT = os.path.join(_PREV_OUT_DIR, "data", f"checkpoint_{STATE_SLUG}.npy")

os.makedirs(DATA_DIR,     exist_ok=True)
os.makedirs(STATES_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)
os.makedirs(ASSETS_DIR,   exist_ok=True)
os.makedirs(LOGS_DIR,     exist_ok=True)
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# Output paths — all variables, never hardcoded strings
# PNGs → ASSETS_DIR  |  CSVs → OUTPUT_DIR  |  logs → LOGS_DIR  |  .npy/.pkl → DATA_OUT_DIR
MAP_PNG        = os.path.join(ASSETS_DIR, f"districts_{STATE_SLUG}_{VERSION}.png")
SUMMARY_PNG    = os.path.join(ASSETS_DIR, f"summary_{STATE_SLUG}_{VERSION}.png")
COMPLEXITY_PNG = os.path.join(ASSETS_DIR, f"census_complexity_{STATE_SLUG}_{VERSION}.png")
BLOCKS_PNG     = os.path.join(ASSETS_DIR, f"census_blocks_{STATE_SLUG}_{VERSION}.png")
SWAP_PNG       = os.path.join(ASSETS_DIR, f"swap_comparison_{STATE_SLUG}_{VERSION}.png")
SWAP_ZOOM_PNG  = os.path.join(ASSETS_DIR, f"swap_zoom_{STATE_SLUG}_{VERSION}.png")
FILLED_PNG     = os.path.join(ASSETS_DIR, f"districts_filled_{STATE_SLUG}_{VERSION}.png")
REPORT_PDF     = os.path.join(OUTPUT_DIR, f"report_{STATE_SLUG}_{VERSION}.pdf")
CODE_PDF       = os.path.join(OUTPUT_DIR, f"source_code_{VERSION}.pdf")

def process_page_path(n: int) -> str:
    return os.path.join(ASSETS_DIR, f"process_page{n}_{STATE_SLUG}_{VERSION}.png")

SPLIT_COLORS = {0:"#C0392B",1:"#E67E22",2:"#27AE60",3:"#8E44AD",4:"#2980B9"}
SPLIT_WIDTH  = {0:2.8,1:2.2,2:1.8,3:1.4,4:1.2}

# Output DPI — separate values let us tune file size vs. quality per output type
MAP_DPI     = 150  # district map and summary chart
BLOCKS_DPI  = 130  # census block explainer (many dots; lower DPI still sharp)
PROCESS_DPI = 140  # step-by-step process pages

# Census block scatter dot size — small enough not to overlap at state scale
SCATTER_SIZE = 0.3

# Geographic / geometric constants
EARTH_RADIUS_KM  = 6371.0  # mean Earth radius for haversine distance
LINE_EXTEND_M    = 2e6     # half-length (m) for infinite split line before clipping to region
DISTRICT_SIMPLIFY_M = 5000  # 5 km smoothing — kept for process-page vector maps only

# Raster visualization constants — Stage 8 district map
MAP_RASTER_W    = 1500   # grid width  (pixels)
MAP_RASTER_H    = 900    # grid height (pixels)
MAP_RASTER_GAMMA     = 0.8   # brightness gamma: <1 = convex, rural cells stay visible
MAP_BRIGHT_FLOOR     = 0.35  # minimum brightness for any populated raster cell
MAP_VORONOI_DIM      = 0.20  # brightness for unpopulated Voronoi fill cells

# ── Logging ───────────────────────────────────────────────────
log_path = os.path.join(LOGS_DIR,
    f"run_{STATE_SLUG}_{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, mode="w")])
log = logging.getLogger()
log.info("=" * 60)
log.info(f"Population-Bisecting Splitline Redistricting {VERSION} — {STATE_NAME}")
log.info(f"  Prepared by: {AUTHOR}")
log.info(f"  Districts: {N_DISTRICTS} | Depth: {MAX_DEPTH} | "
         f"Per-split tol: {MAX_SPLIT_ERROR*100:.3f}%")
log.info("=" * 60)

if not FULL_REPORT:
    log.info("  Mode: quick  (district map PNG only; use --full for reports)")


# ── Stage 1: Load ─────────────────────────────────────────────
log.info("[Stage 1] Loading census block centroids")
URL = (f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
       f"tl_2020_{STATE_FIPS}_tabblock20.zip")

if not os.path.exists(SHP_DIR):
    log.info(f"  Downloading {STATE_NAME} shapefile...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH,"r") as z: z.extractall(SHP_DIR)

shp = [f for f in os.listdir(SHP_DIR) if f.endswith(".shp")][0]
shp_path = os.path.join(SHP_DIR, shp)

# Select projection: Albers Equal-Area for contiguous US, UTM for AK/HI
log.info("  Loading shapefile...")
gdf = gpd.read_file(shp_path)
if STATE_FIPS == "02":    # Alaska
    UTM_EPSG = 3338       # NAD83 Alaska Albers
elif STATE_FIPS == "15":  # Hawaii
    UTM_EPSG = 26904      # NAD83 UTM zone 4N
else:
    UTM_EPSG = 5070       # NAD83 Conus Albers Equal-Area
log.info(f"  Projection: EPSG:{UTM_EPSG}")

gdf = gdf.to_crs(epsg=UTM_EPSG)
gdf["POP20"] = gdf["POP20"].astype(int)
gdf = gdf[(gdf["ALAND20"]>0)&(gdf["POP20"]>0)].copy().reset_index(drop=True)
gdf["cx"] = gdf.geometry.centroid.x
gdf["cy"] = gdf.geometry.centroid.y

_use_cache = False
if os.path.exists(CACHE_CSV):
    _cdf = pd.read_csv(CACHE_CSV)
    if "utm_epsg" in _cdf.columns and int(_cdf["utm_epsg"].iloc[0]) == UTM_EPSG:
        log.info(f"  Cache: {CACHE_CSV}")
        cx_all  = _cdf["cx"].values;   cy_all  = _cdf["cy"].values
        pop_all = _cdf["POP20"].values; geoids  = _cdf["GEOID20"].values
        _use_cache = True
    else:
        log.info(f"  Cache EPSG mismatch — recomputing centroids...")

if not _use_cache:
    cx_all  = gdf["cx"].values;   cy_all  = gdf["cy"].values
    pop_all = gdf["POP20"].values; geoids  = gdf["GEOID20"].values
    gdf[["GEOID20","POP20","cx","cy"]].assign(utm_epsg=UTM_EPSG).to_csv(CACHE_CSV, index=False)
    log.info(f"  Cache saved: {CACHE_CSV}")

total_pop  = pop_all.sum()
target_pop = total_pop / N_DISTRICTS
n_blocks   = len(cx_all)
xmin_s,xmax_s = cx_all.min(),cx_all.max()
ymin_s,ymax_s = cy_all.min(),cy_all.max()
log.info(f"  {n_blocks:,} blocks | pop: {total_pop:,} | target: {target_pop:,.0f}")

utm_to_wgs84 = Transformer.from_crs(f"EPSG:{UTM_EPSG}","EPSG:4326",always_xy=True)
wgs84_to_utm = Transformer.from_crs("EPSG:4326",f"EPSG:{UTM_EPSG}",always_xy=True)

def utm_to_latlon(utm_x: float, utm_y: float) -> tuple[float, float]:
    """Convert projected coordinates to WGS-84 latitude/longitude.

    Args:
        utm_x: Easting in the active projection (meters).
        utm_y: Northing in the active projection (meters).

    Returns:
        (lat, lon) rounded to 4 decimal places.
    """
    lon, lat = utm_to_wgs84.transform(utm_x, utm_y)
    return round(lat, 4), round(lon, 4)


def latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS-84 latitude/longitude to projected coordinates.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        (x, y) easting/northing in the active projection (meters).
    """
    x, y = wgs84_to_utm.transform(lon, lat)
    return x, y


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points in kilometers.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def representative_city(
    district_geom,
    centroid_utm_x: float,
    centroid_utm_y: float,
) -> tuple[str, float, float, float, float, float, bool]:
    """Select the best representative office city for a district.

    Uses Option A: centroid-proximity score (log-population × distance decay).
    All cities inside the district boundary are evaluated; the highest-scoring
    one is returned.  Falls back to the nearest outside city only when no
    qualified city exists inside the boundary.

    Score formula:
        score = log(city_population) × exp(−distance / district_radius)

    where distance is the Euclidean UTM distance from the city to the
    population-weighted centroid, and district_radius = sqrt(area / π) —
    the radius of a circle with the same area as the district.

    This balances two competing goals: pick a city large enough to host a
    congressional office, but keep it near where the district's people live.
    A city twice the district radius away must be roughly e² ≈ 7× more
    populous than the centroid city to outscore it.  A city at the centroid
    exactly scores the full log(population) with no penalty.

    Args:
        district_geom: Shapely geometry of the district in EPSG:5070.
        centroid_utm_x: Population-weighted centroid easting (metres).
        centroid_utm_y: Population-weighted centroid northing (metres).

    Returns:
        (city_name, dist_km, centroid_lat, centroid_lon,
         city_lat, city_lon, city_in_district)
    """
    centroid_lat, centroid_lon = utm_to_latlon(centroid_utm_x, centroid_utm_y)
    district_radius_m = math.sqrt(district_geom.area / math.pi)

    best_score = -1.0
    best_inside: tuple | None = None

    fallback_dist_m = float("inf")
    fallback: tuple | None = None

    for city, clat, clon, cpop in STATE_CITIES:
        city_utm_x, city_utm_y = latlon_to_utm(clat, clon)
        dist_m = math.hypot(city_utm_x - centroid_utm_x, city_utm_y - centroid_utm_y)

        if district_geom.contains(Point(city_utm_x, city_utm_y)):
            score = math.log(max(cpop, 1)) * math.exp(-dist_m / district_radius_m)
            if score > best_score:
                best_score = score
                best_inside = (city, round(dist_m / 1000, 1),
                               centroid_lat, centroid_lon, clat, clon, True)
        else:
            if dist_m < fallback_dist_m:
                fallback_dist_m = dist_m
                fallback = (city, round(dist_m / 1000, 1),
                            centroid_lat, centroid_lon, clat, clon, False)

    if best_inside is not None:
        return best_inside
    return fallback or ("Unknown", 0.0, centroid_lat, centroid_lon,
                        centroid_lat, centroid_lon, False)


def load_tiger_state_boundary(fips: str, target_crs) -> gpd.GeoSeries:
    """Return the official Census TIGER state boundary reprojected to target_crs.

    Downloads tl_2020_us_state.zip once into data/states/ and caches the
    extracted shapefile.  Subsequent calls skip the download.

    Args:
        fips: Two-digit state FIPS code (e.g. "08" for Colorado).
        target_crs: CRS to reproject into — must match the census block data.

    Returns:
        GeoSeries with one Polygon or MultiPolygon for the state boundary.
    """
    os.makedirs(STATES_DIR, exist_ok=True)
    if not os.path.exists(TIGER_STATE_SHP):
        log.info("  Downloading Census TIGER state boundaries...")
        urllib.request.urlretrieve(TIGER_STATE_URL, TIGER_STATE_ZIP)
        with zipfile.ZipFile(TIGER_STATE_ZIP) as z:
            z.extractall(STATES_DIR)
        log.info(f"  Extracted to {STATES_DIR}/")
    states = gpd.read_file(TIGER_STATE_SHP)
    state  = states[states["STATEFP"] == fips]
    return gpd.GeoSeries(state.geometry.values, crs=states.crs).to_crs(target_crs)


# ── Stage 2: Split functions ──────────────────────────────────
def weighted_centroid(indices: np.ndarray) -> tuple[float, float]:
    """Compute the population-weighted centroid of a set of census blocks.

    Args:
        indices: Block indices into the global cx_all / cy_all / pop_all arrays.

    Returns:
        (x, y) centroid in the active projection (meters).
    """
    w = pop_all[indices]
    return (np.average(cx_all[indices], weights=w),
            np.average(cy_all[indices], weights=w))


def balance_at_angle(
    indices: np.ndarray,
    angle_rad: float,
    ax: float,
    ay: float,
    target_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Split blocks along a line and measure how close the population ratio is to target.

    Args:
        indices: Block indices to split.
        angle_rad: Angle of the split line in radians (0 = east, π/2 = north).
        ax: X coordinate of the anchor point (population-weighted centroid).
        ay: Y coordinate of the anchor point.
        target_frac: Desired fraction of total population on the left side.

    Returns:
        (left_mask, right_mask, err, pop_left, pop_right) where err is in [0, 1];
        0 means the split exactly hit target_frac.
    """
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    signed = (cx_all[indices] - ax) * (-dy) + (cy_all[indices] - ay) * dx
    lm = signed <= 0
    rm = ~lm
    pl = pop_all[indices][lm].sum()
    pr = pop_all[indices][rm].sum()
    total = pl + pr
    # Normalized to [0, 1]: 0 = perfect split at target_frac, 1 = all on one side
    err = abs(pl / total - target_frac) * 2 if total > 0 else 1.0
    return lm, rm, err, int(pl), int(pr)


def goal_seek_angle(
    indices: np.ndarray,
    ax: float,
    ay: float,
    seed_angle_deg: float,
    target_frac: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Find the split-line angle that best achieves the target population ratio.

    Phase 1 — bidirectional goal-seek: steps delta from 0 to SEARCH_RADIUS.
    At each delta, both seed+delta and seed-delta are tried.  The first angle
    that achieves <= MAX_SPLIT_ERROR is returned immediately, so the result is
    always the angle closest to the seed that meets the tolerance.

    Phase 2 — full brute-force fallback: if no angle within ±SEARCH_RADIUS works,
    sweeps 0°→179° in order and returns the best found (stopping early if tolerance
    is met).  This fallback fires frequently on irregular sub-regions.

    Args:
        indices: Block indices for this sub-region.
        ax: Anchor X (population-weighted centroid of sub-region).
        ay: Anchor Y.
        seed_angle_deg: Starting angle in degrees to sweep from.
        target_frac: Fraction of population that should go to the left side.

    Returns:
        (angle_rad, left_mask, right_mask, best_err, (pop_left, pop_right))
    """
    best_err=np.inf; best_angle=np.deg2rad(seed_angle_deg)
    best_left=best_right=None; best_pops=(0,0)
    for delta in range(0,int(SEARCH_RADIUS)+1):
        for sign in ([0] if delta==0 else [1,-1]):
            ang=(seed_angle_deg+sign*delta)%180
            ar=np.deg2rad(ang)
            lm,rm,err,pl,pr=balance_at_angle(indices,ar,ax,ay,target_frac)
            if err<best_err:
                best_err=err; best_angle=ar
                best_left=lm; best_right=rm; best_pops=(pl,pr)
            if err<=MAX_SPLIT_ERROR:
                return best_angle,best_left,best_right,best_err,best_pops
    log.info(f"    [!] Expanding to full 180° (best: {best_err*100:.3f}%)")
    for deg in range(0,180):
        ar=np.deg2rad(deg)
        lm,rm,err,pl,pr=balance_at_angle(indices,ar,ax,ay,target_frac)
        if err<best_err:
            best_err=err; best_angle=ar
            best_left=lm; best_right=rm; best_pops=(pl,pr)
        if err<=MAX_SPLIT_ERROR: break
    return best_angle,best_left,best_right,best_err,best_pops

def make_clipped_line(
    ax: float,
    ay: float,
    angle_rad: float,
    region_shape: Any,
) -> Any:
    """Build a split line clipped to a sub-region boundary.

    Extends the line LINE_EXTEND_M meters in both directions from the anchor
    point, then clips it to the region polygon so only the visible segment
    is stored.

    Args:
        ax: Anchor X coordinate (population-weighted centroid).
        ay: Anchor Y coordinate.
        angle_rad: Split-line direction in radians.
        region_shape: Shapely geometry of the sub-region to clip against.

    Returns:
        Clipped Shapely geometry (LineString or MultiLineString), or the
        full unclipped line if intersection raises an exception.
    """
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    full = LineString([
        (ax - LINE_EXTEND_M * dx, ay - LINE_EXTEND_M * dy),
        (ax + LINE_EXTEND_M * dx, ay + LINE_EXTEND_M * dy),
    ])
    try:
        return full.intersection(region_shape)
    except Exception:
        return full


def region_shape_from_indices(indices: np.ndarray) -> Any:
    """Return the dissolved geometry for a subset of census blocks.

    Args:
        indices: Block indices into the global GeoDataFrame.

    Returns:
        Shapely geometry (unary union of the selected block polygons).
    """
    return unary_union(gdf.iloc[indices].geometry.values)


# ── Stage 3: Splitline (with checkpoint) ─────────────────────
# Also check prev version's root dir (pre-v15 stored checkpoints there, not in data/)
_prev_ckpt_root = os.path.join(_PREV_OUT_DIR, f"checkpoint_{STATE_SLUG}.npy")
_ckpt_path = (CHECKPOINT if os.path.exists(CHECKPOINT)
              else _PREV_CHECKPOINT if os.path.exists(_PREV_CHECKPOINT)
              else _prev_ckpt_root)
if os.path.exists(_ckpt_path):
    log.info(f"[Stage 3] Loading checkpoint: {_ckpt_path}")
    ckpt    = np.load(_ckpt_path, allow_pickle=True).item()
    labels  = ckpt["labels"]
    # split_log can't be fully reconstructed from checkpoint (geometry objects)
    # so we rebuild a simplified version for visualization
    split_log_data = ckpt["split_log_data"]
    log.info(f"  Loaded {len(split_log_data)} splits from checkpoint.")

    gdf["district"] = labels
    _need_geom_rebuild = any("clipped_wkt" not in sd for sd in split_log_data)
    if _need_geom_rebuild:
        log.info("  Rebuilding split geometries (one-time; will cache in checkpoint)...")

    split_log = []
    for sd in split_log_data:
        indices   = np.where(np.isin(np.arange(n_blocks), sd["indices_list"]))[0]
        left_mask = np.isin(np.arange(len(indices)),
                            [i for i,idx in enumerate(indices)
                             if idx in sd["left_indices_set"]])
        wkt = sd.get("clipped_wkt")
        if wkt:
            clipped = wkt_loads(wkt)
        else:
            ax,ay = sd["anchor_x"], sd["anchor_y"]
            ar    = np.deg2rad(sd["angle_deg"])
            try:
                region_geom = region_shape_from_indices(indices)
                clipped = make_clipped_line(ax, ay, ar, region_geom)
            except Exception:
                clipped = None
        split_log.append({**sd,
                          "indices": indices,
                          "left_mask": left_mask,
                          "clipped_line": clipped})

    if _need_geom_rebuild:
        for i, (s, sd) in enumerate(zip(split_log, split_log_data)):
            cl = s["clipped_line"]
            split_log_data[i]["clipped_wkt"] = cl.wkt if (cl and not cl.is_empty) else None
        np.save(CHECKPOINT, {"labels": labels, "split_log_data": split_log_data},
                allow_pickle=True)
        log.info("  Checkpoint updated with cached geometry.")
    log.info("  Checkpoint loaded successfully.")
else:
    log.info("[Stage 3] Running population-bisecting splitline")
    labels        = np.full(n_blocks,-1,dtype=int)
    split_log     = []
    split_counter = [0]

    def splitline(indices, n_left_d, n_right_d, d_start, level=0, region_geom=None):
        """
        Recursively bisect a set of census blocks into equal-population districts.
        Each cut passes through the population-weighted centroid of the sub-region.
        Seed angles alternate between 135° (NW-SE) and 45° (NE-SW) to prevent
        successive cuts from all running in the same direction and pinwheeling
        districts around the population center.  The seed is a goal-seek starting
        point — the algorithm sweeps ±SEARCH_RADIUS degrees to find the angle that
        best balances population, so the actual cut rarely lands at exactly 135° or 45°.
        """
        n_total = n_left_d+n_right_d
        if n_total==1: labels[indices]=d_start; return
        if len(indices)==0: return

        ax,ay     = weighted_centroid(indices)
        split_num = split_counter[0]
        seed      = FIRST_CUT_ANGLE if split_num%2==0 else ALT_CUT_ANGLE
        split_counter[0] += 1
        target_frac = n_left_d / n_total  # proportional split, not always 50:50

        angle_rad,left_mask,right_mask,err,(pl,pr) = \
            goal_seek_angle(indices,ax,ay,seed,target_frac)
        angle_deg = np.rad2deg(angle_rad)%180

        if region_geom is None:
            region_geom = region_shape_from_indices(indices)
        clipped = make_clipped_line(ax,ay,angle_rad,region_geom)

        log.info(f"  Split {split_num+1} | L{level} | "
                 f"seed {seed:.0f}° → {angle_deg:.1f}° | "
                 f"err {err*100:.4f}% | L:{pl:,} R:{pr:,}")

        split_log.append({
            "split_num":   split_num+1,
            "level":       level,
            "seed_angle":  seed,
            "angle_deg":   round(angle_deg,1),
            "balance_err": round(err*100,4),
            "anchor_x":    ax, "anchor_y": ay,
            "clipped_line":clipped,
            "pop_left":    pl, "pop_right": pr,
            "d_start":     d_start,
            "n_left":      n_left_d, "n_right": n_right_d,
            "indices":     indices,
            "left_mask":   left_mask,
        })

        nl_l=n_left_d//2;  nl_r=n_left_d-nl_l
        nr_l=n_right_d//2; nr_r=n_right_d-nr_l
        log.info(f"    Computing sub-region geometries...")
        left_geom  = region_shape_from_indices(indices[left_mask])
        right_geom = region_shape_from_indices(indices[right_mask])
        splitline(indices[left_mask], nl_l,nl_r,d_start,          level+1,left_geom)
        splitline(indices[right_mask],nr_l,nr_r,d_start+n_left_d, level+1,right_geom)

    log.info("  Building state geometry...")
    state_geom = region_shape_from_indices(np.arange(n_blocks))
    n_top_l=N_DISTRICTS//2; n_top_r=N_DISTRICTS-n_top_l
    splitline(np.arange(n_blocks),n_top_l,n_top_r,0,level=0,region_geom=state_geom)
    gdf["district"] = labels
    log.info(f"  {len(split_log)} splits completed.")

    # Save checkpoint
    split_log_data = [{k:v for k,v in s.items()
                       if k not in ("indices","left_mask","clipped_line")}
                      for s in split_log]
    for i,s in enumerate(split_log):
        split_log_data[i]["indices_list"]    = s["indices"].tolist()
        split_log_data[i]["left_indices_set"] = set(
            s["indices"][s["left_mask"]].tolist())
        cl = s["clipped_line"]
        split_log_data[i]["clipped_wkt"] = cl.wkt if (cl and not cl.is_empty) else None

    np.save(CHECKPOINT, {"labels": labels,
                          "split_log_data": split_log_data},
            allow_pickle=True)
    log.info(f"  Checkpoint saved: {CHECKPOINT}")


# ── Stage 4: Border-swap ──────────────────────────────────────
SWAP_CHECKPOINT       = os.path.join(DATA_OUT_DIR, f"checkpoint_swap_{STATE_SLUG}.npy")
_PREV_SWAP_CHECKPOINT = os.path.join(_PREV_OUT_DIR, "data", f"checkpoint_swap_{STATE_SLUG}.npy")

_swap_loaded = False
_prev_swap_root = os.path.join(_PREV_OUT_DIR, f"checkpoint_swap_{STATE_SLUG}.npy")
_swap_ckpt_path = (SWAP_CHECKPOINT if os.path.exists(SWAP_CHECKPOINT)
                   else _PREV_SWAP_CHECKPOINT if os.path.exists(_PREV_SWAP_CHECKPOINT)
                   else _prev_swap_root)
if os.path.exists(_swap_ckpt_path):
    log.info(f"[Stage 4] Loading swap checkpoint: {_swap_ckpt_path}")
    labels = np.load(_swap_ckpt_path, allow_pickle=True).item()["labels"]
    gdf["district"] = labels
    _swap_loaded = True
    log.info("  Loaded.")
else:
    log.info("[Stage 4] Border-swap post-processing for <1% deviation")
    pop_stats = gdf.groupby("district")["POP20"].sum()
    dev_pct   = (pop_stats-target_pop)/target_pop*100
    log.info(f"  Before swap: max dev = {dev_pct.abs().max():.4f}%")

    log.info("  Building block adjacency...")
    sindex    = gdf.sindex
    neighbors = [[] for _ in range(n_blocks)]
    for i,geom in enumerate(gdf.geometry):
        for j in list(sindex.intersection(geom.bounds)):
            if i!=j and geom.touches(gdf.geometry.iloc[j]):
                neighbors[i].append(j)

    dist_pops = np.array([pop_all[labels==k].sum()
                          for k in range(N_DISTRICTS)], dtype=float)
    for rnd in range(N_SWAP_ROUNDS):
        cur_dev = float(np.abs(dist_pops-target_pop).max()/target_pop*100)
        if cur_dev<=MAX_TOTAL_DEV*100:
            log.info(f"  Converged at round {rnd+1}: {cur_dev:.4f}%"); break
        swapped=0
        for i in range(n_blocks):
            k=labels[i]
            for m in set(labels[j] for j in neighbors[i] if labels[j]!=k):
                if dist_pops[k]<=target_pop: continue
                if dist_pops[m]>=target_pop: continue
                if dist_pops[k]<=dist_pops[m]: continue
                if not any(labels[j]==k for j in neighbors[i]): continue
                labels[i]=m
                dist_pops[k]-=pop_all[i]; dist_pops[m]+=pop_all[i]
                swapped+=1; break
        if rnd%20==0:
            log.info(f"  Round {rnd+1:3d}: max dev {cur_dev:.4f}% | swapped {swapped}")
        if swapped==0:
            log.info(f"  No more swaps at round {rnd+1}."); break

    gdf["district"] = labels
    np.save(SWAP_CHECKPOINT, {"labels": labels}, allow_pickle=True)
    log.info(f"  Swap checkpoint saved: {SWAP_CHECKPOINT}")

pop_stats = gdf.groupby("district")["POP20"].sum()
dev_pct   = (pop_stats-target_pop)/target_pop*100
max_dev   = dev_pct.abs().max()
result    = "PASS ✓" if max_dev<=MAX_TOTAL_DEV*100 else "FAIL ✗"
log.info(f"  After swap: max dev = {max_dev:.4f}% | {result}")


# ── Stage 5: Dissolve ─────────────────────────────────────────
DISSOLVE_CACHE = os.path.join(DATA_OUT_DIR, f"checkpoint_dissolve_{STATE_SLUG}.pkl")
_prev_dissolve_root = os.path.join(_PREV_OUT_DIR, f"checkpoint_dissolve_{STATE_SLUG}.pkl")
_dissolve_path = (DISSOLVE_CACHE if os.path.exists(DISSOLVE_CACHE)
                  else os.path.join(_PREV_OUT_DIR, "data", f"checkpoint_dissolve_{STATE_SLUG}.pkl")
                  if os.path.exists(os.path.join(_PREV_OUT_DIR, "data", f"checkpoint_dissolve_{STATE_SLUG}.pkl"))
                  else _prev_dissolve_root)
if _swap_loaded and os.path.exists(_dissolve_path):
    log.info("[Stage 5] Loading cached district shapes...")
    with open(_dissolve_path, "rb") as _f:
        district_shapes = pickle.load(_f)
else:
    log.info("[Stage 5] Dissolving district shapes...")
    district_shapes = gdf.dissolve(by="district")
    with open(DISSOLVE_CACHE, "wb") as _f:
        pickle.dump(district_shapes, _f)
    log.info("  Dissolve cache saved.")
hull = district_shapes.geometry.convex_hull
pp   = 4*np.pi*hull.area/hull.length**2
log.info(f"  Avg compactness: {pp.mean():.3f}")


# ── Stage 6: City lookup ──────────────────────────────────────
log.info("[Stage 6] Nearest city lookup")

summary_rows = []
city_outside_warnings = []
for d in range(N_DISTRICTS):
    idx=np.where(labels==d)[0]; w=pop_all[idx]
    cx=np.average(cx_all[idx],weights=w)
    cy=np.average(cy_all[idx],weights=w)
    city,dist_km,lat,lon,city_lat,city_lon,city_in_district = representative_city(
        district_shapes.geometry.loc[d], cx, cy)
    city_utm_x, city_utm_y = latlon_to_utm(city_lat, city_lon)
    ok="✓" if abs(dev_pct[d])<=MAX_TOTAL_DEV*100 else "✗"
    outside_flag = "" if city_in_district else "  ⚠ no city inside district — fallback"
    log.info(f"  D{d+1:<3} {pop_stats[d]:>10,} {dev_pct[d]:>+7.3f}%  "
             f"{city:<22} {dist_km:>5.0f}km  {ok}{outside_flag}")
    if not city_in_district:
        city_outside_warnings.append(f"D{d+1}: {city}")
    area_km2 = district_shapes.geometry.loc[d].area / 1e6   # m² → km²
    summary_rows.append({
        "district":d+1, "population":int(pop_stats[d]),
        "deviation_pct":round(float(dev_pct[d]),4),
        "within_1pct":abs(dev_pct[d])<=MAX_TOTAL_DEV*100,
        "pp_compactness":round(float(pp.iloc[d]),4),
        "area_km2":round(area_km2, 1),
        "pop_density_per_km2":round(pop_stats[d] / area_km2, 2),
        "center_lat":lat, "center_lon":lon,
        "center_utm_x":cx, "center_utm_y":cy,
        "nearest_city":city, "nearest_city_dist_km":dist_km,
        "city_utm_x":city_utm_x, "city_utm_y":city_utm_y,
        "city_in_district":city_in_district,
    })

if city_outside_warnings:
    log.warning(f"  ⚠ No city inside district boundary (fallback used): {', '.join(city_outside_warnings)}")
summary_df = pd.DataFrame(summary_rows)



def build_raster_state_mask(state_geometry, query_xy: np.ndarray,
                             shape: tuple,
                             cache_path: str | None = None) -> np.ndarray:
    """Return a boolean raster mask: True where a cell centre is inside the state.

    Tries shapely 2.x vectorized contains (fast GEOS STRtree), falls back to
    matplotlib Path ray-casting.  Cached to cache_path if provided so subsequent
    runs (same state, same grid size) skip the computation entirely.

    Args:
        state_geometry: Shapely Polygon or MultiPolygon for the state.
        query_xy: (N, 2) array of (x, y) cell-centre coordinates.
        shape: (H, W) output grid shape.
        cache_path: Optional .npy file path for caching the result.

    Returns:
        Boolean ndarray of shape (H, W).
    """
    if cache_path and os.path.exists(cache_path):
        log.info(f"  Using cached state mask: {cache_path}")
        return np.load(cache_path)

    try:
        import shapely as _shp
        pts    = _shp.points(query_xy[:, 0], query_xy[:, 1])
        inside = _shp.contains(state_geometry, pts).reshape(shape)
    except Exception:
        # Fallback: matplotlib Path ray-casting (slower on complex polygons)
        polys = list(state_geometry.geoms) if state_geometry.geom_type == "MultiPolygon" \
                else [state_geometry]
        flat = np.zeros(len(query_xy), dtype=bool)
        for poly in polys:
            flat |= MplPath(np.array(poly.exterior.coords)).contains_points(query_xy)
            for interior in poly.interiors:
                flat &= ~MplPath(np.array(interior.coords)).contains_points(query_xy)
        inside = flat.reshape(shape)

    if cache_path:
        np.save(cache_path, inside)
    return inside


# ── Drawing helpers ───────────────────────────────────────────
def set_state_bounds(ax: Any, pad: float = 15000) -> None:
    ax.set_xlim(xmin_s - pad, xmax_s + pad)
    ax.set_ylim(ymin_s - pad, ymax_s + pad)
    ax.set_axis_off()


def draw_clipped_splits(ax: Any, splits: list[dict]) -> None:
    for s in splits:
        col = SPLIT_COLORS.get(s["level"], "#555")
        wid = SPLIT_WIDTH.get(s["level"], 1.0)
        line = s["clipped_line"]
        if line is None or line.is_empty:
            continue
        geoms = [line] if line.geom_type == "LineString" else list(line.geoms)
        for g in geoms:
            if g.geom_type != "LineString":
                continue
            xs, ys = g.xy
            ax.plot(xs, ys, color=col, linewidth=wid * 1.8,
                    linestyle="--", alpha=0.92, zorder=6)
        ax.plot(s["anchor_x"], s["anchor_y"], "o", color=col, markersize=8,
                zorder=7, markeredgecolor="white", markeredgewidth=1.5)


def draw_region_labels(ax: Any, s: dict, fontsize: int = 11) -> None:
    for idx_set, pop_val in [
        (s["indices"][s["left_mask"]],   s["pop_left"]),
        (s["indices"][~s["left_mask"]], s["pop_right"]),
    ]:
        if len(idx_set) == 0:
            continue
        w = pop_all[idx_set]
        lx = np.average(cx_all[idx_set], weights=w)
        ly = np.average(cy_all[idx_set], weights=w)
        ax.annotate(f"{pop_val / 1000:.0f}k", (lx, ly),
                    ha="center", va="center", fontsize=fontsize, fontweight="500",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.90, edgecolor="#AAAAAA", linewidth=0.5), zorder=9)


def split_legend_handles(include_dot: bool = True) -> list:
    levels = sorted(set(s["level"] for s in split_log))
    handles = [
        Line2D([0], [0], color=SPLIT_COLORS.get(l, "#555"),
               linewidth=SPLIT_WIDTH.get(l, 1) * 1.8, linestyle="--",
               label=f"Level {l} cuts")
        for l in levels
    ]
    if include_dot:
        handles.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="#555555", markersize=9,
                               markeredgecolor="white", markeredgewidth=1.2,
                               label="● Population center of sub-region\n"
                                     "  (bisection anchor point)"))
    return handles


# ── Stage 8: Final district map ───────────────────────────────
log.info("[Stage 8] Final district map...")

MAP_BG = "#0D1117"

# Vivid colour for each district — boosts saturation/value for dark backgrounds
def _vivid(hex_color: str) -> tuple[float, float, float]:
    """Return a dark-background-safe version of hex_color (S≥0.75, V≥0.80)."""
    h, s, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(hex_color))
    return colorsys.hsv_to_rgb(h, max(s, 0.75), max(v, 0.80))

_district_rgb = np.array([_vivid(MAP_COLORS[d % len(MAP_COLORS)]) for d in range(N_DISTRICTS)])

# ── Official state boundary from Census TIGER ─────────────────────────────
log.info("  Loading official state boundary from Census TIGER...")
_state_gs   = load_tiger_state_boundary(STATE_FIPS, district_shapes.crs)
_state_poly = _state_gs.geometry.values[0]
log.info(f"  State boundary: {_state_poly.geom_type}")

# ── Raster grid ───────────────────────────────────────────────────────────
_W, _H = MAP_RASTER_W, MAP_RASTER_H
_x_bins = np.linspace(xmin_s, xmax_s, _W + 1)
_y_bins = np.linspace(ymin_s, ymax_s, _H + 1)
_xc = (_x_bins[:-1] + _x_bins[1:]) / 2
_yc = (_y_bins[:-1] + _y_bins[1:]) / 2
_XX, _YY   = np.meshgrid(_xc, _yc)                           # (_H, _W)
_query_pts = np.column_stack([_XX.ravel(), _YY.ravel()])     # (_H*_W, 2)

# Per-district population grids → city-lights background
_pixel_rgb = np.zeros((_H, _W, 3))
_pixel_pop = np.zeros((_H, _W))
for _d in range(N_DISTRICTS):
    _dm = (labels == _d) & (pop_all > 0)
    if not _dm.any():
        continue
    _g, _, _ = np.histogram2d(cx_all[_dm], cy_all[_dm],
                               bins=[_x_bins, _y_bins], weights=pop_all[_dm])
    _g    = _g.T
    _beat = _g > _pixel_pop
    _pixel_pop        = np.where(_beat, _g, _pixel_pop)
    _pixel_rgb[_beat] = _district_rgb[_d]

def _brightness(pop_grid: np.ndarray) -> np.ndarray:
    lp = np.log10(np.maximum(pop_grid, 1))
    lo, hi = lp.min(), lp.max()
    b = np.clip(((lp - lo) / max(hi - lo, 1e-9)) ** MAP_RASTER_GAMMA, 0.0, 1.0)
    return np.where(pop_grid > 0, np.maximum(b, MAP_BRIGHT_FLOOR), 0.0)

_pixel_rgb_lit = _pixel_rgb * _brightness(_pixel_pop)[:, :, np.newaxis]

# ── Voronoi district grid + state mask ────────────────────────────────────
log.info("  Building Voronoi district grid...")
_tree = cKDTree(np.column_stack([cx_all, cy_all]))
_, _nn = _tree.query(_query_pts, k=1, workers=-1)
_vd   = labels[_nn].reshape(_H, _W)         # every cell → nearest block's district

log.info("  Building state raster mask...")
_mask_cache = os.path.join(STATES_DIR,
    f"state_mask_{STATE_FIPS}_{_W}x{_H}.npy")
_in_state = build_raster_state_mask(_state_poly, _query_pts, (_H, _W), _mask_cache)
log.info(f"  In-state: {_in_state.sum():,}/{_H*_W:,} cells "
         f"({100*_in_state.mean():.1f}%)")

# Remove enclosed district islands from the Voronoi grid using hole-filling.
# After border swaps, some census blocks end up geographically isolated inside a
# different district — their Voronoi cells form closed loops on the boundary layer.
# binary_fill_holes closes any region completely surrounded by a single district,
# regardless of size.  Compute all fills from the original _vd first, then apply
# once to avoid ordering artifacts between districts.
from scipy.ndimage import binary_fill_holes as _ndi_fill_holes
_fill_target = np.full((_H, _W), -1, dtype=np.intp)
_fill_count   = 0
for _d in range(N_DISTRICTS):
    _holes = _ndi_fill_holes(_vd == _d) & (_vd != _d) & _in_state
    if _holes.any():
        _fill_target[_holes] = _d
        _fill_count += int(_holes.sum())
if _fill_count:
    _vd = _vd.copy()
    _vd[_fill_target >= 0] = _fill_target[_fill_target >= 0]
log.info(f"  Voronoi hole-fill: {_fill_count:,} enclosed pixels reassigned")

# District boundary pixels: 4-connected Voronoi neighbours differ, clipped to state.
_ch = _vd[:, :-1] != _vd[:, 1:]
_cv = _vd[:-1, :] != _vd[1:, :]
_bnd = np.zeros((_H, _W), dtype=bool)
_bnd[:, :-1] |= _ch;  _bnd[:, 1:]  |= _ch
_bnd[:-1, :] |= _cv;  _bnd[1:, :]  |= _cv
_bnd &= _in_state                            # clip — no lines escape the state border
log.info(f"  Boundary pixels: {_bnd.sum():,}")

# Each boundary pixel gets the vivid colour of its district
_bnd_rgba = np.zeros((_H, _W, 4))
_bnd_rgba[_bnd, :3] = _district_rgb[_vd[_bnd]]
_bnd_rgba[_bnd, 3]  = 1.0

# ── Render ────────────────────────────────────────────────────────────────
# Figure width is derived from geographic extent so the state is never keystoned.
_fig_h_map = 10.0
_fig_w_map = _fig_h_map * (xmax_s - xmin_s) / (ymax_s - ymin_s)
fig, ax = plt.subplots(1, 1, figsize=(_fig_w_map, _fig_h_map))
fig.patch.set_facecolor(MAP_BG)
ax.set_facecolor(MAP_BG)

# Layer 1: city-lights (population density coloured by district)
ax.imshow(_pixel_rgb_lit, origin="lower",
          extent=[xmin_s, xmax_s, ymin_s, ymax_s],
          aspect="equal", interpolation="nearest", zorder=2)
# Layer 2: coloured district boundary lines (rasterised, no interior islands)
ax.imshow(_bnd_rgba, origin="lower",
          extent=[xmin_s, xmax_s, ymin_s, ymax_s],
          aspect="equal", interpolation="nearest", zorder=3)
# Layer 3: official state border
_state_gs.boundary.plot(ax=ax, color="white", linewidth=2.0, zorder=4)

# External callout labels: stars on the map at city locations, text labels pulled
# outside the state border with connecting lines.
#
# Label placement uses "radial-angle" positioning: each label is placed on the
# margin rectangle at the same angle as its city from the state centre.  A line
# from a label to its own radial direction is purely radial, so two such lines
# can never cross (radial lines through a common centre don't intersect).
# When cities cluster at similar angles the labels would overlap, so a 1-D
# relaxation pass spreads them apart while preserving CCW order.
#
# Duplicate cities: when N districts share the same nearest city, their stars
# would overlap and only one would be visible.  We detect this and offset each
# star to a small circle so all N are individually visible.
_sc_x = (xmin_s + xmax_s) / 2
_sc_y = (ymin_s + ymax_s) / 2
_sw_m = xmax_s - xmin_s
_sh_m = ymax_s - ymin_s
_lb_px = _sw_m * 0.17          # horizontal margin for label zone
_lb_py = _sh_m * 0.22          # vertical margin
_LX0 = xmin_s - _lb_px;  _LX1 = xmax_s + _lb_px
_LY0 = ymin_s - _lb_py;  _LY1 = ymax_s + _lb_py
_LHW = (_LX1 - _LX0) / 2;  _LHH = (_LY1 - _LY0) / 2

def _box_pt(theta: float) -> tuple[float, float]:
    """Intersection of ray at angle theta from label-box centre with box perimeter."""
    c, s = math.cos(theta), math.sin(theta)
    if abs(c) < 1e-9:
        return _sc_x, _sc_y + math.copysign(_LHH, s)
    if abs(s) < 1e-9:
        return _sc_x + math.copysign(_LHW, c), _sc_y
    t = min(_LHW / abs(c), _LHH / abs(s))
    return _sc_x + t * c, _sc_y + t * s

# ── Step 1: jitter stars for duplicate cities ─────────────────────────────
# Group districts by rounded city coordinates; offset stars in a small circle
# so every district gets a distinct visible star.
_STAR_JITTER_M = 12_000   # 12 km jitter radius
_star_xy: dict[int, tuple[float, float]] = {}
_coord_groups: dict[tuple, list] = {}
for _d in range(N_DISTRICTS):
    _key = (round(summary_rows[_d]['city_utm_x'], -3),
            round(summary_rows[_d]['city_utm_y'], -3))
    _coord_groups.setdefault(_key, []).append(_d)
for _key, _ds in _coord_groups.items():
    _bx = summary_rows[_ds[0]]['city_utm_x']
    _by = summary_rows[_ds[0]]['city_utm_y']
    if len(_ds) == 1:
        _star_xy[_ds[0]] = (_bx, _by)
    else:
        for _ji, _d in enumerate(_ds):
            _ja = math.pi / 2 + 2 * math.pi * _ji / len(_ds)   # start at top, go CCW
            _star_xy[_d] = (_bx + _STAR_JITTER_M * math.cos(_ja),
                            _by + _STAR_JITTER_M * math.sin(_ja))

# ── Step 2: radial-angle label placement with 1-D relaxation ─────────────
# Compute each city's radial angle from the state bounding-box centre.
_city_theta = [
    (math.atan2(summary_rows[_d]['city_utm_y'] - _sc_y,
                summary_rows[_d]['city_utm_x'] - _sc_x) + 2 * math.pi) % (2 * math.pi)
    for _d in range(N_DISTRICTS)
]
# Sort districts CCW by that angle.
_sorted_d = sorted(range(N_DISTRICTS), key=lambda _d: _city_theta[_d])

# ── Pass 1: bidirectional angular relaxation ─────────────────────────────────
# Initialise label angles at each city's radial angle, then push adjacent pairs
# apart symmetrically (half-deficit each direction) until all gaps >= min_gap.
# Bidirectional spreading distributes cluster overlap evenly rather than
# front-loading it — fixes collisions when several cities share the same bearing.
_min_gap = 2 * math.pi / N_DISTRICTS * 0.60
_lbl_a = [_city_theta[_d] for _d in _sorted_d]
for _ in range(500):
    _moved = False
    for _i in range(N_DISTRICTS):
        _j = (_i + 1) % N_DISTRICTS
        _gap = (_lbl_a[_j] - _lbl_a[_i]) % (2 * math.pi)
        if _gap < _min_gap:
            _half = (_min_gap - _gap) / 2
            _lbl_a[_i] = (_lbl_a[_i] - _half) % (2 * math.pi)
            _lbl_a[_j] = (_lbl_a[_j] + _half) % (2 * math.pi)
            _moved = True
    if not _moved:
        break

# Re-sort label angles back into the city CCW order.
# Bidirectional relaxation can push a label past its neighbour (especially
# when several cities cluster in a narrow arc and the backward push wraps
# past another label).  Sorting the produced angles and re-assigning them to
# the city-sorted districts restores the invariant that label positions are in
# the same CCW order as city positions — which by the non-crossing theorem
# guarantees no connecting lines cross.
_lbl_a = sorted(_lbl_a)

_lbl_fs = max(5.0, min(8.0, 70.0 / N_DISTRICTS))

# ── Pass 2: coordinate-space bounding-box overlap check ──────────────────────
# Angular separation alone doesn't guarantee visual separation — two labels on
# the same edge of the margin rectangle can be angularly apart but still
# overlapping if their text is long.  For each adjacent pair, estimate the
# label box dimensions in map units, detect any box overlap, and nudge angles
# apart (bidirectionally) until the boxes clear.
_m_per_pt_x = (_LX1 - _LX0) / (_fig_w_map * 72)   # map metres per typographic point
_m_per_pt_y = (_LY1 - _LY0) / (_fig_h_map * 72)

def _label_box(idx: int) -> tuple[float, float]:
    """Half-width and half-height of label box in map units."""
    _d = _sorted_d[idx]
    _txt = f"D{_d+1} · {summary_rows[_d]['nearest_city']}"
    return (len(_txt) * _lbl_fs * 0.55 * _m_per_pt_x / 2,
            _lbl_fs * 1.6  * _m_per_pt_y / 2)

for _ in range(200):
    _moved = False
    for _i in range(N_DISTRICTS):
        _j = (_i + 1) % N_DISTRICTS
        _xi, _yi = _box_pt(_lbl_a[_i])
        _xj, _yj = _box_pt(_lbl_a[_j])
        _hwi, _hhi = _label_box(_i)
        _hwj, _hhj = _label_box(_j)
        _ox = (_hwi + _hwj) - abs(_xj - _xi)   # x overlap (positive = overlap)
        _oy = (_hhi + _hhj) - abs(_yj - _yi)   # y overlap
        if _ox > 0 and _oy > 0:
            # Numerically estimate how much angular change eliminates the tighter overlap.
            _EPS = 5e-4
            _xi2, _yi2 = _box_pt((_lbl_a[_i] - _EPS) % (2 * math.pi))
            _xj2, _yj2 = _box_pt((_lbl_a[_j] + _EPS) % (2 * math.pi))
            _gain_x = (abs(_xj2 - _xi2) - abs(_xj - _xi)) / _EPS   # metres per radian
            _gain_y = (abs(_yj2 - _yi2) - abs(_yj - _yi)) / _EPS
            if abs(_gain_x) >= abs(_gain_y) and abs(_gain_x) > 1.0:
                _nudge = _ox / abs(_gain_x)
            elif abs(_gain_y) > 1.0:
                _nudge = _oy / abs(_gain_y)
            else:
                _nudge = 0.008
            _nudge = max(0.003, min(_nudge, 0.04))
            _lbl_a[_i] = (_lbl_a[_i] - _nudge) % (2 * math.pi)
            _lbl_a[_j] = (_lbl_a[_j] + _nudge) % (2 * math.pi)
            _moved = True
    if not _moved:
        break

# Re-sort after Pass 2 for the same reason as after Pass 1.
_lbl_a = sorted(_lbl_a)

for _i, _d in enumerate(_sorted_d):
    _row     = summary_rows[_d]
    _d_color = MAP_COLORS[_d % len(MAP_COLORS)]
    _csx, _csy = _star_xy[_d]

    # Star at (possibly jittered) city location
    ax.plot(_csx, _csy, marker='*', linestyle='none', color='black', markersize=14, zorder=10)
    ax.plot(_csx, _csy, marker='*', linestyle='none', color=_d_color, markersize=11,
            markeredgecolor='white', markeredgewidth=0.7, zorder=11)

    # Label on the margin rectangle at the relaxed radial angle
    _lx, _ly = _box_pt(_lbl_a[_i])
    _on_r = _lx > _LX0 + (_LX1 - _LX0) * 0.92
    _on_l = _lx < _LX0 + (_LX1 - _LX0) * 0.08
    _ha = 'left' if _on_r else ('right' if _on_l else 'center')
    _va = 'center' if (_on_r or _on_l) else ('bottom' if _ly > _sc_y else 'top')

    ax.plot([_lx, _csx], [_ly, _csy], color='white', linewidth=0.5, alpha=0.45, zorder=9)
    ax.annotate(
        f"D{_d+1} · {_row['nearest_city']}",
        (_lx, _ly), ha=_ha, va=_va, fontsize=_lbl_fs, fontweight="500", color="white",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#1A1A2E",
                  alpha=0.80, edgecolor=_d_color, linewidth=0.5), zorder=14)

_dist_label = "District" if N_DISTRICTS == 1 else "Districts"
ax.set_title(
    f"{STATE_NAME} — {N_DISTRICTS} {_dist_label} ({APPORTIONMENT_YEAR} apportionment)\n"
    f"Population-Bisecting Splitline {VERSION}  ·  "
    f"Colour = District  ·  Brightness = population density",
    fontsize=12, fontweight="500", color="white", pad=12)
# Expand view to include the label margin zone
ax.set_xlim(_LX0 - _lb_px * 0.1, _LX1 + _lb_px * 0.1)
ax.set_ylim(_LY0 - _lb_py * 0.1, _LY1 + _lb_py * 0.1)

plt.suptitle(
    f"{STATE_NAME} Population-Bisecting Splitline Redistricting {VERSION}  |  "
    f"Pop: {total_pop:,}  |  Target: {target_pop:,.0f}/district  |  "
    f"Max deviation: {max_dev:.4f}%  |  {result}",
    fontsize=11, fontweight="500", color="white", y=1.01)
plt.tight_layout()
plt.savefig(MAP_PNG, dpi=MAP_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
log.info(f"  Saved: {MAP_PNG}")



# ── Quick-mode exit (also covers single-district states) ─────
if not FULL_REPORT or N_DISTRICTS == 1:
    log.info("=" * 60)
    log.info(f"Done — {STATE_NAME} {VERSION}")
    log.info(f"  Prepared by:     {AUTHOR}")
    log.info(f"  Districts:       {N_DISTRICTS}")
    log.info(f"  Max deviation:   {max_dev:.4f}% | {result}")
    log.info(f"  Avg compactness: {pp.mean():.3f}")
    log.info(f"  District map:    {MAP_PNG}")
    log.info("=" * 60)
    sys.exit(0)


# ── Stage 7: Census block explainer ───────────────────────────
log.info("[Stage 7] Census block explainer...")
gdf["area_sqkm"] = gdf["ALAND20"] / 1e6
block_areas = gdf["area_sqkm"].values
block_pops  = gdf["POP20"].values
blk_stats = {
    "n_blocks":    len(gdf),
    "total_pop":   int(total_pop),
    "area_mean":   block_areas.mean(),
    "area_median": float(np.median(block_areas)),
    "area_min":    block_areas.min(),
    "area_max":    block_areas.max(),
    "pop_min":     int(block_pops.min()),
    "pop_mean":    float(block_pops.mean()),
    "pop_max":     int(block_pops.max()),
    "pop_median":  float(np.median(block_pops)),
}

# ── Stage 7a: Census block complexity map ─────────────────────
# Draw every census block as a faint white outline on a dark background.
# In cities where blocks are tiny and packed, thousands of outlines overlap
# and the area glows white.  In rural areas the few large blocks barely show.
# This map is ONLY about data structure — no district colors, no population.
if os.path.exists(COMPLEXITY_PNG):
    log.info(f"  [7a] Using cached: {COMPLEXITY_PNG}")
else:
    log.info(f"  [7a] Building census block complexity map ({n_blocks:,} blocks)...")
    fig_cx, ax_cx = plt.subplots(figsize=(16, 10))
    fig_cx.patch.set_facecolor("#1A1A2E")
    ax_cx.set_facecolor("#1A1A2E")
    gdf.plot(ax=ax_cx, facecolor="none", edgecolor="white",
             linewidth=0.15, alpha=0.20)
    ax_cx.set_title(
        f"{STATE_NAME} — {n_blocks:,} Census Block Boundaries\n"
        "Each line is one block edge  ·  Lines stack in cities to reveal density",
        fontsize=14, fontweight="500", color="white", pad=10)
    set_state_bounds(ax_cx, pad=20000)
    plt.tight_layout()
    fig_cx.savefig(COMPLEXITY_PNG, dpi=BLOCKS_DPI, bbox_inches="tight",
                   facecolor=fig_cx.get_facecolor())
    plt.close()
    log.info(f"  Saved: {COMPLEXITY_PNG}")

# ── Stage 7b: Population density map ─────────────────────────
if os.path.exists(BLOCKS_PNG):
    log.info(f"  [7b] Using cached: {BLOCKS_PNG}")
else:
    log.info(f"  [7b] Building population density map...")

    # Auto-pick three inset areas: big city, mid-size city, rural
    _sw = xmax_s - xmin_s
    _sh = ymax_s - ymin_s
    _uhw = _sw * 0.03
    _rhw = _sw * 0.07

    _city_utms = []
    for _ic, _ilat, _ilon, *_ in STATE_CITIES[:min(30, len(STATE_CITIES))]:
        _iux, _iuy = latlon_to_utm(_ilat, _ilon)
        _city_utms.append((_ic, _iux, _iuy))

    _A_name, _A_x, _A_y = _city_utms[0]
    _B_name, _B_x, _B_y = _city_utms[min(1, len(_city_utms) - 1)]
    for _ic, _iux, _iuy in _city_utms[1:]:
        if math.sqrt((_iux - _A_x) ** 2 + (_iuy - _A_y) ** 2) > _sw * 0.20:
            _B_name, _B_x, _B_y = _ic, _iux, _iuy
            break

    _all_cux = [x for _, x, _ in _city_utms]
    _all_cuy = [y for _, _, y in _city_utms]
    _blk_cx  = gdf["cx"].values
    _blk_cy  = gdf["cy"].values

    # Only consider rural candidates where census blocks actually exist in the
    # inset region — bounding-box corners can be outside the state boundary.
    _best_rd = 0
    _R_x, _R_y = (xmin_s + xmax_s) / 2, (ymin_s + ymax_s) / 2
    for _gi in range(25):
        for _gj in range(20):
            _gx = xmin_s + (_gi + 0.5) * _sw / 25
            _gy = ymin_s + (_gj + 0.5) * _sh / 20
            _has_blks = (
                (_blk_cx >= _gx - _rhw) & (_blk_cx <= _gx + _rhw) &
                (_blk_cy >= _gy - _rhw) & (_blk_cy <= _gy + _rhw)
            ).any()
            if not _has_blks:
                continue
            _md = min(math.sqrt((_gx - _ccx) ** 2 + (_gy - _ccy) ** 2)
                      for _ccx, _ccy in zip(_all_cux, _all_cuy))
            if _md > _best_rd:
                _best_rd = _md
                _R_x, _R_y = _gx, _gy

    def _clamp_box(cx, cy, hw):
        pad = hw * 0.1
        return (max(xmin_s + pad, cx - hw), min(xmax_s - pad, cx + hw),
                max(ymin_s + pad, cy - hw), min(ymax_s - pad, cy + hw))

    A_X0, A_X1, A_Y0, A_Y1 = _clamp_box(_A_x, _A_y, _uhw)
    B_X0, B_X1, B_Y0, B_Y1 = _clamp_box(_B_x, _B_y, _uhw)
    # Fit rural box to the actual extent of blocks in the region, not the
    # full search radius — otherwise irregular state shapes leave the inset mostly empty.
    _r_in_rgn = (
        (_blk_cx >= _R_x - _rhw) & (_blk_cx <= _R_x + _rhw) &
        (_blk_cy >= _R_y - _rhw) & (_blk_cy <= _R_y + _rhw)
    )
    if _r_in_rgn.any():
        _r_bx = _blk_cx[_r_in_rgn];  _r_by = _blk_cy[_r_in_rgn]
        _r_hw = max(
            (_r_bx.max() - _r_bx.min()) * 0.58,
            (_r_by.max() - _r_by.min()) * 0.58,
            15_000,   # 15 km minimum so a few whole blocks are always visible
        )
        R_X0, R_X1, R_Y0, R_Y1 = _clamp_box(
            float((_r_bx.max() + _r_bx.min()) / 2),
            float((_r_by.max() + _r_by.min()) / 2),
            _r_hw)
    else:
        R_X0, R_X1, R_Y0, R_Y1 = _clamp_box(_R_x, _R_y, _rhw)
    log.info(f"  Insets: [A] {_A_name}  [B] {_B_name}  [C] Rural area")

    fig_b, ax_b = plt.subplots(figsize=(16, 10))
    fig_b.patch.set_facecolor("#1A1A2E")
    ax_b.set_facecolor("#1A1A2E")
    pop_vals = np.clip(gdf["POP20"].values, 1, None)
    log_pop  = np.log10(pop_vals)
    norm_b   = mcolors.Normalize(vmin=log_pop.min(), vmax=log_pop.max())
    cmap_b   = plt.cm.YlOrRd
    ax_b.scatter(gdf["cx"].values, gdf["cy"].values,
                 c=log_pop, cmap=cmap_b, norm=norm_b,
                 s=SCATTER_SIZE, alpha=0.6, linewidths=0, zorder=2)

    inset_defs_main = [
        (A_X0, A_Y0, A_X1, A_Y1, _A_name, "A", "#00D4FF"),
        (B_X0, B_Y0, B_X1, B_Y1, _B_name, "B", "#FFD700"),
        (R_X0, R_Y0, R_X1, R_Y1, "Rural Area", "C", "#98FF98"),
    ]
    for (x0, y0, x1, y1, lbl, letter, col) in inset_defs_main:
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   linewidth=2, edgecolor=col, facecolor="none", zorder=8)
        ax_b.add_patch(rect)
        _label_gap = (y1 - y0) * 0.08   # offset scales with the box, not hardcoded to city size
        ax_b.annotate(f"[{letter}] {lbl}", (x0 + (x1 - x0) / 2, y1 + _label_gap),
                      ha="center", va="bottom", fontsize=9,
                      color=col, fontweight="bold", zorder=9)

    sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
    sm_b.set_array([])
    cbar_b = fig_b.colorbar(sm_b, ax=ax_b, shrink=0.45, pad=0.06,
                             label="Population (log scale)")
    cbar_b.set_ticks([1, 2, 3, 4])
    cbar_b.set_ticklabels(["10", "100", "1,000", "10,000"])
    cbar_b.ax.yaxis.label.set_color("white")
    cbar_b.ax.tick_params(colors="white")
    ax_b.set_title(
        f"{STATE_NAME} — All {blk_stats['n_blocks']:,} Census Blocks\n"
        "Each dot = one block, colored by population (brighter = more people)",
        fontsize=14, fontweight="500", color="white", pad=10)
    set_state_bounds(ax_b, pad=20000)

    stats_text = (
        f"Census Block Summary\n{'─' * 26}\n"
        f"Total blocks: {blk_stats['n_blocks']:>8,}\n"
        f"Total pop:    {blk_stats['total_pop']:>8,}\n\n"
        f"Area (km²)\n"
        f" Min:    {blk_stats['area_min']:>8.4f}\n"
        f" Median: {blk_stats['area_median']:>8.2f}\n"
        f" Mean:   {blk_stats['area_mean']:>8.2f}\n"
        f" Max:    {blk_stats['area_max']:>8.0f}\n\n"
        f"Population\n"
        f" Min:    {blk_stats['pop_min']:>8,}\n"
        f" Median: {blk_stats['pop_median']:>8.0f}\n"
        f" Mean:   {blk_stats['pop_mean']:>8.1f}\n"
        f" Max:    {blk_stats['pop_max']:>8,}"
    )
    ax_b.text(0.02, 0.02, stats_text, transform=ax_b.transAxes,
              fontsize=8, verticalalignment="bottom", fontfamily="monospace",
              color="white",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="#0D1B2A",
                        alpha=0.88, edgecolor="#444"))
    plt.tight_layout(rect=[0, 0, 0.83, 1])

    blocks_main_tmp = os.path.join(OUTPUT_DIR, "blocks_main_tmp.png")
    fig_b.savefig(blocks_main_tmp, dpi=BLOCKS_DPI, bbox_inches="tight",
                  facecolor=fig_b.get_facecolor())
    plt.close()

    fig_b2, axes_b2 = plt.subplots(1, 3, figsize=(16, 5))
    fig_b2.patch.set_facecolor("#1A1A2E")
    inset_detail = [
        (A_X0, A_Y0, A_X1, A_Y1, f"[A] {_A_name}", "#00D4FF"),
        (B_X0, B_Y0, B_X1, B_Y1, f"[B] {_B_name}", "#FFD700"),
        (R_X0, R_Y0, R_X1, R_Y1, "[C] Rural Area",  "#98FF98"),
    ]
    for ax_i, (x0, y0, x1, y1, ititle, icol) in zip(axes_b2, inset_detail):
        ax_i.set_facecolor("#0D1B2A")
        imask = (gdf["cx"] >= x0) & (gdf["cx"] <= x1) & (gdf["cy"] >= y0) & (gdf["cy"] <= y1)
        isub = gdf[imask]
        if len(isub) > 0:
            ilp   = np.log10(np.clip(isub["POP20"].values, 1, None))
            inorm = mcolors.Normalize(vmin=ilp.min(), vmax=max(ilp.max(), 2))
            for geom, ip in zip(isub.geometry, ilp):
                try:
                    xs2, ys2 = geom.exterior.xy
                    ax_i.fill(xs2, ys2, color=cmap_b(inorm(ip)), alpha=0.8)
                    ax_i.plot(xs2, ys2, color="#333333", linewidth=0.2, alpha=0.5)
                except Exception:
                    pass
            for _, rs in isub.nlargest(3, "POP20").iterrows():
                ax_i.annotate(
                    f"Pop: {int(rs['POP20']):,}\n{rs['area_sqkm']:.2f} km²",
                    (rs["cx"], rs["cy"]), ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#000",
                              alpha=0.7, edgecolor="none"), zorder=10)
            ax_i.set_title(
                f"{ititle}\n{len(isub):,} blocks | "
                f"{int(isub['POP20'].sum()):,} people | "
                f"median {np.median(isub['area_sqkm']):.2f} km²",
                fontsize=9, color="white", fontweight="500", pad=6)
        ax_i.set_xlim(x0, x1)
        ax_i.set_ylim(y0, y1)
        for spine in ax_i.spines.values():
            spine.set_edgecolor(icol)
            spine.set_linewidth(2)
        ax_i.set_xticks([])
        ax_i.set_yticks([])

    plt.suptitle(
        "Inset Detail: Urban Density vs Rural Sparsity — same data, different scales",
        fontsize=11, color="white", fontweight="500", y=1.02)
    plt.tight_layout()
    blocks_insets_tmp = os.path.join(OUTPUT_DIR, "blocks_insets_tmp.png")
    fig_b2.savefig(blocks_insets_tmp, dpi=BLOCKS_DPI, bbox_inches="tight",
                   facecolor=fig_b2.get_facecolor())
    plt.close()

    # ── Histogram: block population distribution ─────────────────
    _pop_all  = gdf["POP20"].values
    _h_edges  = [0, 1, 10, 50, 200, 500, 1_000, 2_500, 5_000, 10_000, np.inf]
    _h_labels = ["0", "1–9", "10–49", "50–199", "200–499", "500–999",
                 "1K–2.5K", "2.5K–5K", "5K–10K", "10K+"]
    _h_counts = np.array([
        int(((_pop_all >= _h_edges[i]) & (_pop_all < _h_edges[i + 1])).sum())
        for i in range(len(_h_labels))
    ])
    _h_pops = np.array([
        int(_pop_all[(_pop_all >= _h_edges[i]) & (_pop_all < _h_edges[i + 1])].sum())
        for i in range(len(_h_labels))
    ])
    _h_cum = _h_pops.cumsum()
    _h_total = int(_pop_all.sum())

    fig_hist, ax_hb = plt.subplots(figsize=(16, 4.5))
    fig_hist.patch.set_facecolor("#1A1A2E")
    ax_hb.set_facecolor("#0D1B2A")
    _hx = np.arange(len(_h_labels))
    _bar_cols = [cmap_b(0.15 + 0.80 * i / max(len(_h_labels) - 1, 1))
                 for i in range(len(_h_labels))]
    ax_hb.bar(_hx, _h_counts, color=_bar_cols, edgecolor="#0D1B2A", linewidth=0.5, zorder=3)
    ax_hb.set_ylabel("Number of Blocks", color="white", fontsize=10)
    ax_hb.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{int(v/1_000)}K" if v >= 1_000 else str(int(v))))
    ax_hb.tick_params(colors="white")
    ax_hb.set_xticks(_hx)
    ax_hb.set_xticklabels(_h_labels, color="white", fontsize=10)
    ax_hb.set_xlabel("Block Population", color="white", fontsize=10)
    ax_hb.grid(axis="y", color="#1A2030", linewidth=0.8, zorder=0)
    for sp in ax_hb.spines.values():
        sp.set_edgecolor("#333355")

    ax_hr = ax_hb.twinx()
    ax_hr.plot(_hx, _h_cum, color="#00D4FF", linewidth=2.5,
               marker="o", markersize=6, zorder=5)
    ax_hr.fill_between(_hx, _h_cum, alpha=0.10, color="#00D4FF")
    ax_hr.set_ylim(0, _h_total * 1.08)
    ax_hr.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v/1_000_000:.1f}M" if v >= 1_000_000 else f"{v/1_000:.0f}K"))
    ax_hr.set_ylabel("Cumulative Population", color="#00D4FF", fontsize=10)
    ax_hr.tick_params(colors="#00D4FF")
    # Annotation at 50% and 100% cumulative
    for _pct, _col, _lbl in [(0.50, "#FFD700", "50%"), (1.00, "#00D4FF", "100%")]:
        _tgt = _h_total * _pct
        ax_hr.axhline(_tgt, color=_col, linewidth=0.9, linestyle="--", alpha=0.55)
        ax_hr.annotate(f"{_lbl}  {int(_tgt):,}", (_hx[-1] - 0.1, _tgt),
                       ha="right", va="bottom", color=_col, fontsize=8)

    ax_hb.set_title(
        f"Block Population Distribution — {blk_stats['n_blocks']:,} blocks total  ·  "
        f"Bars = block count per range  ·  Blue line = cumulative population",
        fontsize=10, color="white", fontweight="500", pad=8)
    plt.tight_layout()
    blocks_hist_tmp = os.path.join(OUTPUT_DIR, "blocks_hist_tmp.png")
    fig_hist.savefig(blocks_hist_tmp, dpi=BLOCKS_DPI, bbox_inches="tight",
                     facecolor=fig_hist.get_facecolor())
    plt.close()

    img_main   = PILImage.open(blocks_main_tmp)
    img_insets = PILImage.open(blocks_insets_tmp)
    img_hist   = PILImage.open(blocks_hist_tmp)
    w_combined = max(img_main.width, img_insets.width, img_hist.width)
    h_combined = img_main.height + img_insets.height + img_hist.height
    combined   = PILImage.new("RGB", (w_combined, h_combined), "#1A1A2E")
    combined.paste(img_main,   (0, 0))
    combined.paste(img_insets, (0, img_main.height))
    combined.paste(img_hist,   (0, img_main.height + img_insets.height))
    combined.save(BLOCKS_PNG)
    os.remove(blocks_main_tmp)
    os.remove(blocks_insets_tmp)
    os.remove(blocks_hist_tmp)
    log.info(f"  Saved: {BLOCKS_PNG}")


# ── Stage 9: Summary table ────────────────────────────────────
log.info("[Stage 9] Summary table...")
fig3, ax4 = plt.subplots(1, 1, figsize=(10, 0.5 + 0.6 * N_DISTRICTS))
fig3.patch.set_facecolor("#F8F8F5")
ax4.set_facecolor("#F8F8F5")
ax4.set_axis_off()
td = [["District", "Population", "Deviation", "Rep. Office Area", "Compactness"]]
for r in summary_rows:
    ok = "✓" if r["within_1pct"] else "✗"
    td.append([f"D{r['district']}", f"{r['population']:,}",
               f"{r['deviation_pct']:+.3f}% {ok}",
               f"{r['nearest_city']} ({r['nearest_city_dist_km']:.0f}km)",
               f"{r['pp_compactness']:.3f}"])
t4 = ax4.table(cellText=td[1:], colLabels=td[0], cellLoc="center", loc="center",
               colWidths=[0.12, 0.20, 0.20, 0.34, 0.14])
t4.auto_set_font_size(False); t4.set_fontsize(10); t4.scale(1, 2.0)
for (r4, c4), cell in t4.get_celld().items():
    if r4 == 0:
        cell.set_facecolor("#1B3A5C"); cell.set_text_props(color="white", fontweight="bold")
    elif r4 % 2 == 0:
        cell.set_facecolor("#F0F4F8")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#DDDDDD")
ax4.set_title("District Summary", fontsize=12, fontweight="500", pad=12)
plt.suptitle(f"{STATE_NAME} Population-Bisecting Splitline Redistricting {VERSION}  |  "
             f"Max deviation: {max_dev:.4f}% | {result}",
             fontsize=11, fontweight="500", y=1.02)
plt.tight_layout()
plt.savefig(SUMMARY_PNG, dpi=MAP_DPI, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
log.info(f"  Saved: {SUMMARY_PNG}")


# ── Stage 9b: Swap comparison visualization ───────────────────
log.info("[Stage 9b] Swap comparison visualization...")
_need_swap_png = not os.path.exists(SWAP_PNG)
_need_zoom_png = not os.path.exists(SWAP_ZOOM_PNG)
if not (_need_swap_png or _need_zoom_png):
    log.info(f"  Using cached: {SWAP_PNG} and {SWAP_ZOOM_PNG}")
else:
    # Pre-swap labels from Stage 3 checkpoint (written before Stage 4 border swaps)
    _pre_labels = np.load(CHECKPOINT, allow_pickle=True).item()["labels"]
    _swapped    = _pre_labels != labels
    _n_swapped  = int(_swapped.sum())
    log.info(f"  Blocks reassigned by swap: {_n_swapped:,} of {n_blocks:,}")

    # Build pre-swap Voronoi raster, reusing Stage 8 KDTree nearest-neighbour map
    _vd_pre = _pre_labels[_nn].reshape(_H, _W)
    _fill_target_pre = np.full((_H, _W), -1, dtype=np.intp)
    for _d in range(N_DISTRICTS):
        _holes_pre = _ndi_fill_holes(_vd_pre == _d) & (_vd_pre != _d) & _in_state
        if _holes_pre.any():
            _fill_target_pre[_holes_pre] = _d
    if (_fill_target_pre >= 0).any():
        _vd_pre = _vd_pre.copy()
        _vd_pre[_fill_target_pre >= 0] = _fill_target_pre[_fill_target_pre >= 0]

    # Pre-swap city-lights raster: histogram2d from pre-swap labels
    _pixel_rgb_pre = np.zeros((_H, _W, 3))
    _pixel_pop_pre = np.zeros((_H, _W))
    for _d in range(N_DISTRICTS):
        _dm_pre = (_pre_labels == _d) & (pop_all > 0)
        if not _dm_pre.any():
            continue
        _g_pre, _, _ = np.histogram2d(
            cx_all[_dm_pre], cy_all[_dm_pre],
            bins=[_x_bins, _y_bins], weights=pop_all[_dm_pre])
        _g_pre = _g_pre.T
        _beat_pre = _g_pre > _pixel_pop_pre
        _pixel_pop_pre             = np.where(_beat_pre, _g_pre, _pixel_pop_pre)
        _pixel_rgb_pre[_beat_pre]  = _district_rgb[_d]
    _pixel_rgb_lit_pre = _pixel_rgb_pre * _brightness(_pixel_pop_pre)[:, :, np.newaxis]

    # Pre-swap coloured district boundary lines
    _ch_pre = _vd_pre[:, :-1] != _vd_pre[:, 1:]
    _cv_pre = _vd_pre[:-1, :] != _vd_pre[1:, :]
    _bnd_pre = np.zeros((_H, _W), dtype=bool)
    _bnd_pre[:, :-1] |= _ch_pre;  _bnd_pre[:, 1:]  |= _ch_pre
    _bnd_pre[:-1, :] |= _cv_pre;  _bnd_pre[1:, :]  |= _cv_pre
    _bnd_pre &= _in_state
    _bnd_rgba_pre = np.zeros((_H, _W, 4))
    _bnd_rgba_pre[_bnd_pre, :3] = _district_rgb[_vd_pre[_bnd_pre]]
    _bnd_rgba_pre[_bnd_pre, 3]  = 1.0

    if _need_swap_png:
        # State-level side-by-side comparison — dark background, geographic aspect
        _fig_h_sw = 10.0
        _fig_w_sw = _fig_h_sw * (xmax_s - xmin_s) / (ymax_s - ymin_s)
        fig_sw, axes_sw = plt.subplots(1, 2, figsize=(2 * _fig_w_sw + 0.3, _fig_h_sw))
        fig_sw.patch.set_facecolor(MAP_BG)

        _ax_l = axes_sw[0]
        _ax_l.set_facecolor(MAP_BG)
        _ax_l.imshow(_pixel_rgb_lit_pre, origin="lower",
                     extent=[xmin_s, xmax_s, ymin_s, ymax_s],
                     aspect="equal", interpolation="nearest", zorder=2)
        _ax_l.imshow(_bnd_rgba_pre, origin="lower",
                     extent=[xmin_s, xmax_s, ymin_s, ymax_s],
                     aspect="equal", interpolation="nearest", zorder=3)
        _state_gs.boundary.plot(ax=_ax_l, color="white", linewidth=1.5, zorder=4)
        draw_clipped_splits(_ax_l, split_log)
        set_state_bounds(_ax_l)
        _ax_l.set_axis_off()
        _ax_l.set_title(
            "Splitlines Only\nStraight geometric cuts before census-block refinement",
            fontsize=11, fontweight="500", color="white", pad=8)

        _ax_r = axes_sw[1]
        _ax_r.set_facecolor(MAP_BG)
        _ax_r.imshow(_pixel_rgb_lit, origin="lower",
                     extent=[xmin_s, xmax_s, ymin_s, ymax_s],
                     aspect="equal", interpolation="nearest", zorder=2)
        _ax_r.imshow(_bnd_rgba, origin="lower",
                     extent=[xmin_s, xmax_s, ymin_s, ymax_s],
                     aspect="equal", interpolation="nearest", zorder=3)
        _state_gs.boundary.plot(ax=_ax_r, color="white", linewidth=1.5, zorder=4)
        set_state_bounds(_ax_r)
        _ax_r.set_axis_off()
        _ax_r.set_title(
            "After Border-Swap Refinement\nFinal boundaries follow census block edges",
            fontsize=11, fontweight="500", color="white", pad=8)

        plt.suptitle(
            f"{STATE_NAME} — Border-Swap Refinement  |  "
            f"{_n_swapped:,} of {n_blocks:,} blocks reassigned",
            fontsize=11, fontweight="500", color="white", y=1.01)
        plt.tight_layout()
        plt.savefig(SWAP_PNG, dpi=MAP_DPI, bbox_inches="tight",
                    facecolor=fig_sw.get_facecolor())
        plt.close()
        log.info(f"  Saved: {SWAP_PNG}")

    if _need_zoom_png:
        # Zoom into the 60km grid cell with the most swapped blocks
        _zoom_cell = 60_000  # metres
        _xi_g = np.floor((cx_all - xmin_s) / _zoom_cell).astype(int)
        _yi_g = np.floor((cy_all - ymin_s) / _zoom_cell).astype(int)
        _cell_keys = _xi_g * 100000 + _yi_g
        _sw_counts: dict[int, int] = {}
        for _ck in _cell_keys[_swapped]:
            _sw_counts[_ck] = _sw_counts.get(_ck, 0) + 1
        if _sw_counts:
            _best_ck = max(_sw_counts, key=_sw_counts.get)
            _bxi_g, _byi_g = _best_ck // 100000, _best_ck % 100000
            _zcx = xmin_s + (_bxi_g + 0.5) * _zoom_cell
            _zcy = ymin_s + (_byi_g + 0.5) * _zoom_cell
            log.info(f"  Zoom centre: ({_zcx/1000:.0f}, {_zcy/1000:.0f}) km  "
                     f"({_sw_counts[_best_ck]} swapped blocks in peak cell)")
        else:
            _zcx = float(np.median(cx_all[_swapped]))
            _zcy = float(np.median(cy_all[_swapped]))
        _zr = 80_000   # 80 km radius — shows enough context around the swap zone
        _zx0, _zx1 = _zcx - _zr, _zcx + _zr
        _zy0, _zy1 = _zcy - _zr, _zcy + _zr

        fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(10, 10))
        fig_zoom.patch.set_facecolor(MAP_BG)
        ax_zoom.set_facecolor(MAP_BG)
        ax_zoom.imshow(_pixel_rgb_lit, origin="lower",
                       extent=[xmin_s, xmax_s, ymin_s, ymax_s],
                       aspect="equal", interpolation="nearest", zorder=2)
        ax_zoom.imshow(_bnd_rgba, origin="lower",
                       extent=[xmin_s, xmax_s, ymin_s, ymax_s],
                       aspect="equal", interpolation="nearest", zorder=3)
        _state_gs.boundary.plot(ax=ax_zoom, color="white", linewidth=2.0, zorder=4)
        draw_clipped_splits(ax_zoom, split_log)
        ax_zoom.set_xlim(_zx0, _zx1)
        ax_zoom.set_ylim(_zy0, _zy1)
        ax_zoom.set_axis_off()
        ax_zoom.set_title(
            f"{STATE_NAME} — Splitlines vs. Final Block Boundaries (zoomed)\n"
            "Lines = original bisections  ·  Jagged edges = census block boundaries after swapping",
            fontsize=11, fontweight="500", color="white", pad=8)
        plt.tight_layout()
        plt.savefig(SWAP_ZOOM_PNG, dpi=MAP_DPI, bbox_inches="tight",
                    facecolor=fig_zoom.get_facecolor())
        plt.close()
        log.info(f"  Saved: {SWAP_ZOOM_PNG}")


# ── Stage 10: Process pages ───────────────────────────────────
log.info("[Stage 10] Process pages...")
n_splits=len(split_log); n_pages=math.ceil(n_splits/2)

for page in range(n_pages):
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 11))
    fig2.patch.set_facecolor("#F8F8F5")
    for col in range(2):
        split_idx = page * 2 + col
        ax = axes2[col]
        if split_idx >= n_splits:
            # ── Final state panel ──────────────────────────────────
            ax.set_facecolor("#D6EAF8")
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    district_shapes.loc[[d]].plot(ax=ax,
                        color=MAP_COLORS[d % len(MAP_COLORS)], linewidth=0, alpha=0.90)
            district_shapes.boundary.plot(ax=ax, color="white", linewidth=1.8)
            draw_clipped_splits(ax, split_log)
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    c = district_shapes.loc[d].geometry.centroid
                    ax.annotate(f"D{d+1}\n{pop_stats[d]/1000:.0f}k",
                                (c.x, c.y), ha="center", va="center", fontsize=8,
                                fontweight="500",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                          alpha=0.88, edgecolor="none"), zorder=9)
            ax.set_title(f"Final — {N_DISTRICTS} Districts\n"
                         f"Max deviation: {max_dev:.4f}% | {result}",
                         fontsize=14, fontweight="500", pad=12)
            set_state_bounds(ax)
        else:
            # ── Split panel: zoomed to sub-region ──────────────────
            s = split_log[split_idx]
            splits_so_far = split_log[:split_idx + 1]

            # Compute zoom bounds from this sub-region's block centroids
            rcx = cx_all[s["indices"]]; rcy = cy_all[s["indices"]]
            rpx = max((rcx.max() - rcx.min()) * 0.10, 30000)
            rpy = max((rcy.max() - rcy.min()) * 0.10, 30000)
            zx0, zx1 = rcx.min() - rpx, rcx.max() + rpx
            zy0, zy1 = rcy.min() - rpy, rcy.max() + rpy

            ax.set_facecolor("#D6EAF8")
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    district_shapes.loc[[d]].plot(ax=ax, color="#E8E8E8",
                                                   linewidth=0, alpha=0.5)
            district_shapes.boundary.plot(ax=ax, color="#CCCCCC",
                                          linewidth=0.4, alpha=0.6)
            side_a = s["indices"][s["left_mask"]]
            side_b = s["indices"][~s["left_mask"]]
            ax.scatter(cx_all[side_a], cy_all[side_a], c="#5B9BD5",
                       s=2.5, alpha=0.65, linewidths=0, zorder=2)
            ax.scatter(cx_all[side_b], cy_all[side_b], c="#E8785A",
                       s=2.5, alpha=0.65, linewidths=0, zorder=2)
            draw_clipped_splits(ax, splits_so_far)
            draw_region_labels(ax, s, fontsize=11)
            side_leg = [
                mpatches.Patch(facecolor="#5B9BD5", alpha=0.7,
                               label=f"Side A: {s['pop_left']:,}"),
                mpatches.Patch(facecolor="#E8785A", alpha=0.7,
                               label=f"Side B: {s['pop_right']:,}"),
            ]
            ax.legend(handles=side_leg, loc="upper right",
                      fontsize=8, framealpha=0.9, edgecolor="none")

            seed_dir = "NW–SE" if s["seed_angle"] == FIRST_CUT_ANGLE else "NE–SW"
            landed_note = (f"goal-seek from {s['seed_angle']:.0f}° ({seed_dir}) "
                           f"→ landed {s['angle_deg']:.1f}°"
                           if abs(s["angle_deg"] - s["seed_angle"]) > 0.9
                           else f"{s['angle_deg']:.1f}° (matched seed)")
            ax.set_title(
                f"Split {s['split_num']} of {n_splits}  ·  {landed_note}\n"
                f"Balance error: {s['balance_err']:.3f}%  ·  "
                f"Side A (teal): {s['pop_left']:,}  ·  Side B (salmon): {s['pop_right']:,}\n"
                f"● = population-weighted center of this sub-region (bisection anchor)",
                fontsize=10, fontweight="500", pad=8)

            # Set zoom
            ax.set_xlim(zx0, zx1)
            ax.set_ylim(zy0, zy1)
            ax.set_axis_off()

            # Locator inset — bottom-right, shows where this sub-region sits in the state
            ax_loc = ax.inset_axes([0.77, 0.02, 0.21, 0.21])
            ax_loc.set_facecolor("#D6EAF8")
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    district_shapes.loc[[d]].plot(
                        ax=ax_loc, color="#C0CEDE", linewidth=0, aspect=None)
            # Highlight the sub-region being split
            ax_loc.scatter(rcx, rcy, c="#4A7EB5", s=0.2, linewidths=0, alpha=0.7, zorder=2)
            # Red rectangle showing zoom area
            ax_loc.add_patch(mpatches.Rectangle(
                (zx0, zy0), zx1 - zx0, zy1 - zy0,
                linewidth=1.5, edgecolor="#C0392B", facecolor="#C0392B",
                alpha=0.20, zorder=3, transform=ax_loc.transData))
            ax_loc.set_xlim(xmin_s - 15000, xmax_s + 15000)
            ax_loc.set_ylim(ymin_s - 15000, ymax_s + 15000)
            ax_loc.set_aspect("equal")
            ax_loc.set_axis_off()
            ax_loc.set_title("location", fontsize=6, color="#555555", pad=2)

    axes2[0].legend(handles=split_legend_handles(include_dot=True),
                    loc="lower left", fontsize=8, framealpha=0.92, edgecolor="none")
    plt.suptitle(
        f"{STATE_NAME} Redistricting {VERSION} — Process Page {page + 1} of {n_pages}  |  "
        f"Teal = Side A · Salmon = Side B  |  ● = Population center before bisection",
        fontsize=11, fontweight="500", y=1.01)
    plt.tight_layout()
    plt.savefig(process_page_path(page + 1), dpi=PROCESS_DPI, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.close()
    log.info(f"  Saved: {process_page_path(page + 1)}")

# Standalone solid-color district map — saved separately so it can be viewed directly
if os.path.exists(FILLED_PNG):
    log.info(f"  [10] Using cached: {FILLED_PNG}")
else:
    log.info(f"  [10] Saving standalone filled district map...")
    fig_filled, ax_filled = plt.subplots(figsize=(14, 10))
    fig_filled.patch.set_facecolor("#D6EAF8")
    ax_filled.set_facecolor("#D6EAF8")
    for d in range(N_DISTRICTS):
        if d in district_shapes.index:
            district_shapes.loc[[d]].plot(ax=ax_filled,
                color=MAP_COLORS[d % len(MAP_COLORS)], linewidth=0, alpha=0.90)
    district_shapes.boundary.plot(ax=ax_filled, color="white", linewidth=1.8)
    draw_clipped_splits(ax_filled, split_log)
    for d in range(N_DISTRICTS):
        if d in district_shapes.index:
            c = district_shapes.loc[d].geometry.centroid
            ax_filled.annotate(
                f"D{d+1}\n{pop_stats[d]/1000:.0f}k",
                (c.x, c.y), ha="center", va="center", fontsize=10,
                fontweight="500",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.88, edgecolor="none"), zorder=9)
    ax_filled.set_title(
        f"{STATE_NAME} — {N_DISTRICTS} Final Districts\n"
        f"Max deviation: {max_dev:.4f}% | {result}",
        fontsize=14, fontweight="500", pad=12)
    set_state_bounds(ax_filled)
    plt.tight_layout()
    fig_filled.savefig(FILLED_PNG, dpi=MAP_DPI, bbox_inches="tight",
                       facecolor=fig_filled.get_facecolor())
    plt.close()
    log.info(f"  Saved: {FILLED_PNG}")


# ── Stage 11: CSVs ────────────────────────────────────────────
log.info("[Stage 11] Exporting CSVs...")
gdf[["GEOID20","POP20","district"]].to_csv(
    os.path.join(OUTPUT_DIR, f"block_assignments_{STATE_SLUG}_{VERSION}.csv"),
    index=False)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, f"district_summary_{STATE_SLUG}_{VERSION}.csv"),
    index=False)
pd.DataFrame([{k:v for k,v in s.items()
               if k not in ("indices","left_mask","clipped_line")}
              for s in split_log]).to_csv(
    os.path.join(OUTPUT_DIR, f"split_log_{STATE_SLUG}_{VERSION}.csv"),
    index=False)


# ── Helper: build PDF paragraph styles ───────────────────────
def make_pdf_styles():
    """Return a dict of ReportLab paragraph styles.
    Defined in a function so no local variables can shadow LETTER_SIZE."""
    return {
        "title":   ParagraphStyle("T",fontName="Helvetica-Bold",fontSize=22,
                       textColor=PDF_NAVY,spaceAfter=6,leading=26),
        "sub":     ParagraphStyle("S",fontName="Helvetica",fontSize=12,
                       textColor=PDF_BLUE,spaceAfter=16,leading=16),
        "h1":      ParagraphStyle("H1",fontName="Helvetica-Bold",fontSize=14,
                       textColor=PDF_NAVY,spaceBefore=18,spaceAfter=6,leading=18),
        "h2":      ParagraphStyle("H2",fontName="Helvetica-Bold",fontSize=11,
                       textColor=PDF_BLUE,spaceBefore=12,spaceAfter=4,leading=14),
        "body":    ParagraphStyle("B",fontName="Helvetica",fontSize=10,
                       textColor=PDF_BLACK,leading=15,spaceAfter=8,
                       alignment=TA_JUSTIFY),
        "bullet":  ParagraphStyle("BU",fontName="Helvetica",fontSize=10,
                       textColor=PDF_BLACK,leading=15,spaceAfter=5,leftIndent=20),
        "mono":    ParagraphStyle("M",fontName="Courier",fontSize=9,
                       textColor=HC("#333333"),leading=13,spaceAfter=4,leftIndent=20),
        "foot":    ParagraphStyle("F",fontName="Helvetica",fontSize=8,
                       textColor=HC("#888888"),alignment=TA_CENTER),
        "center":  ParagraphStyle("C",fontName="Helvetica",fontSize=10,
                       textColor=HC("#555555"),alignment=TA_CENTER,spaceAfter=6),
        "code_title": ParagraphStyle("CT",fontName="Helvetica-Bold",
                       fontSize=16,textColor=PDF_NAVY,spaceAfter=6,leading=20),
        "code_sub":   ParagraphStyle("CS",fontName="Helvetica",fontSize=10,
                       textColor=PDF_BLUE,spaceAfter=12),
        "code_mono":  ParagraphStyle("CM",fontName="Courier",fontSize=7.5,
                       leading=11,spaceAfter=0,textColor=HC("#222222")),
        "code_foot":  ParagraphStyle("CF",fontName="Helvetica",fontSize=8,
                       textColor=HC("#888888"),alignment=TA_CENTER),
    }

def escape_xml(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


# ── Stage 12: Source code PDF ─────────────────────────────────
log.info("[Stage 12] Building source code PDF...")
ps = make_pdf_styles()
script_path = os.path.abspath(__file__)
with open(script_path,"r") as fh:
    source_lines = fh.readlines()

code_doc = SimpleDocTemplate(CODE_PDF, pagesize=LETTER_SIZE,
    leftMargin=0.75*inch, rightMargin=0.75*inch,
    topMargin=0.75*inch,  bottomMargin=0.75*inch)

code_story = []
code_story.append(Paragraph(
    f"{STATE_NAME} Population-Bisecting Splitline Redistricting — Full Source Code", ps["code_title"]))
code_story.append(Paragraph(
    f"File: {os.path.basename(script_path)}  |  Version: {VERSION}  |  "
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
    f"Lines: {len(source_lines)}  |  Prepared by: {AUTHOR}",
    ps["code_sub"]))
code_story.append(HRFlowable(width="100%",thickness=1,
                              color=PDF_BLUE,spaceAfter=10))

CHUNK=60
for ci in range(0,len(source_lines),CHUNK):
    chunk=source_lines[ci:ci+CHUNK]
    text="".join(f"{ci+j+1:4d}  {escape_xml(line)}"
                 for j,line in enumerate(chunk))
    text=text.replace("\t","    ").replace("\n","<br/>")
    code_story.append(Paragraph(text, ps["code_mono"]))

code_story.append(Spacer(1,12))
code_story.append(HRFlowable(width="100%",thickness=1,
                              color=PDF_MGRAY,spaceAfter=6))
code_story.append(Paragraph(
    f"{STATE_NAME} Population-Bisecting Splitline Redistricting {VERSION}  |  Full source code  |  "
    f"Prepared by: {AUTHOR}", ps["code_foot"]))

code_doc.build(code_story)
log.info(f"  Saved: {CODE_PDF}")


# ── Stage 13: Main report PDF ─────────────────────────────────
log.info("[Stage 13] Building main report PDF...")

def req_row(num, rtitle, color, text):
    data=[[
        Paragraph(f'<font color="white"><b>{num}</b></font>',
                  ParagraphStyle("N",fontName="Helvetica-Bold",fontSize=14,
                                 textColor=PDF_WHITE,alignment=TA_CENTER)),
        Paragraph(f'<b>{rtitle}</b><br/>{text}',
                  ParagraphStyle("RT",fontName="Helvetica",fontSize=10,
                                 leading=14,textColor=PDF_BLACK))
    ]]
    t=Table(data,colWidths=[0.5*inch,5.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0),color),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(0,0),8),("LEFTPADDING",(1,0),(1,0),12),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[PDF_LGRAY]),
        ("BOX",(0,0),(-1,-1),0.5,PDF_MGRAY),
    ]))
    return t

story=[]

# Compute image embed dimensions from actual PNG sizes so the PDF never keystones.
# max_h_in caps the height so tall-state images never overflow the page body.
def _img_embed(path: str, max_w_in: float, max_h_in: float = 9.0) -> tuple[float, float]:
    """Return (width, height) in ReportLab points fitting within max_w_in × max_h_in inches."""
    pil_w, pil_h = PILImage.open(path).size
    w = max_w_in * inch
    h = w * pil_h / pil_w
    if h > max_h_in * inch:
        h = max_h_in * inch
        w = h * pil_w / pil_h
    return w, h

_map_embed_cover   = _img_embed(MAP_PNG,    6.5, max_h_in=5.5)  # cover — leaves room for stats table
_map_embed_body    = _img_embed(MAP_PNG,    6.8, max_h_in=7.0)  # district map section
_filled_embed      = _img_embed(FILLED_PNG, 6.8, max_h_in=7.0)  # solid-color filled district map

# Cover
story.append(Spacer(1,0.4*inch))
story.append(Paragraph(f"{STATE_NAME}", ps["title"]))
story.append(Paragraph("Population-Bisecting Splitline Districting", ps["sub"]))
story.append(HRFlowable(width="100%",thickness=2,color=PDF_BLUE,spaceAfter=14))
story.append(RLImage(MAP_PNG, width=_map_embed_cover[0], height=_map_embed_cover[1]))
story.append(Spacer(1,14))
story.append(HRFlowable(width="100%",thickness=1,color=PDF_MGRAY,spaceAfter=10))

cover_stats=[
    ("Algorithm","Population-Bisecting Splitline"),
    ("Version", VERSION),
    ("Apportionment Year", str(APPORTIONMENT_YEAR)),
    ("Congressional Districts", str(N_DISTRICTS)),
    ("State Population", f"{total_pop:,}"),
    ("Target per District", f"{target_pop:,.0f}"),
    ("Max Population Deviation", f"{max_dev:.4f}%"),
    ("Result", result),
]
cst_data=[[
    Paragraph(f"<b>{k}</b>",ParagraphStyle("SK",fontName="Helvetica-Bold",
              fontSize=10,textColor=PDF_NAVY)),
    Paragraph(v,ParagraphStyle("SV",fontName="Helvetica",fontSize=10,
              textColor=PDF_BLACK))
] for k,v in cover_stats]
cst=Table(cst_data,colWidths=[2.5*inch,4.0*inch])
cst.setStyle(TableStyle([
    ("ROWBACKGROUNDS",(0,0),(-1,-1),[PDF_LGRAY,PDF_WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),10),("GRID",(0,0),(-1,-1),0.5,PDF_MGRAY),
]))
story.append(cst)
story.append(Spacer(1,18))
story.append(HRFlowable(width="100%",thickness=1,color=PDF_MGRAY,spaceAfter=8))
story.append(Paragraph(f"Prepared by: <b>{AUTHOR}</b>", ps["center"]))
story.append(Paragraph(
    f"Data: {APPORTIONMENT_YEAR} U.S. Census Bureau  |  "
    f"Generated: {datetime.now().strftime('%B %d, %Y')}",ps["center"]))
story.append(Paragraph(
    f"Full source code: {os.path.basename(CODE_PDF)}",ps["center"]))

# Census block explainer
story.append(PageBreak())
story.append(Paragraph("Understanding the Raw Data: Census Blocks",ps["h1"]))
story.append(Paragraph(
    f"The algorithm works with {blk_stats['n_blocks']:,} individual census blocks "
    f"— the smallest geographic unit the Census Bureau counts. Each block is a "
    f"polygon bounded by streets, waterways, or other visible features, containing "
    f"a precise count of residents on Census day (April 1, 2020). District "
    f"boundaries in this proposal always follow block boundaries — they never cut "
    f"through a block.",ps["body"]))
story.append(Paragraph(
    "The first map below draws every block boundary as a faint white line on a dark "
    "background. Where blocks are small and tightly packed — dense urban cores — "
    "thousands of overlapping edges light up the map. Where blocks are large and "
    "sparse — rural plains and mountains — the few edges barely register. This map "
    "shows data structure only: no population, no districts.",ps["body"]))
_complexity_embed = _img_embed(COMPLEXITY_PNG, 6.8, max_h_in=6.5)
story.append(RLImage(COMPLEXITY_PNG, width=_complexity_embed[0], height=_complexity_embed[1]))
story.append(Spacer(1,10))
story.append(Paragraph(
    "The second map colors each block by its population. "
    "Three inset panels compare urban density against rural sparsity at finer detail.",
    ps["body"]))
story.append(RLImage(BLOCKS_PNG,width=6.8*inch,height=5.5*inch))
story.append(Spacer(1,8))

bsd=[["Metric","Value","Notes"],
     ["Total blocks",f"{blk_stats['n_blocks']:,}","Populated only (land>0, pop>0)"],
     ["Total population",f"{blk_stats['total_pop']:,}","2020 Census apportionment"],
     ["Block area — min",f"{blk_stats['area_min']:.4f} km²","Tiny urban blocks"],
     ["Block area — median",f"{blk_stats['area_median']:.2f} km²","Half are smaller"],
     ["Block area — mean",f"{blk_stats['area_mean']:.2f} km²","Pulled up by rural blocks"],
     ["Block area — max",f"{blk_stats['area_max']:.0f} km²","Vast rural/wilderness"],
     ["Block pop — min",f"{blk_stats['pop_min']:,}","Smallest populated block"],
     ["Block pop — median",f"{blk_stats['pop_median']:.0f}","Half have fewer people"],
     ["Block pop — mean",f"{blk_stats['pop_mean']:.1f}","Average per block"],
     ["Block pop — max",f"{blk_stats['pop_max']:,}","Most populous single block"],]
bst=Table(bsd,colWidths=[2.2*inch,1.4*inch,3.1*inch])
bst.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),PDF_NAVY),("TEXTCOLOR",(0,0),(-1,0),PDF_WHITE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[PDF_WHITE,PDF_LGRAY]),
    ("GRID",(0,0),(-1,-1),0.5,PDF_MGRAY),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),8),
]))
story.append(bst)

# Algorithm explainer
story.append(PageBreak())
story.append(Paragraph("The Problem with Gerrymandering",ps["h1"]))
story.append(Paragraph(
    "Every ten years, after the U.S. Census counts everyone, each state redraws "
    "its congressional districts. The people doing the drawing — usually state "
    "legislators — naturally tend to draw lines that help their own party. This "
    "is called gerrymandering, and it's been happening since 1812. The result: "
    "districts shaped like snaking tentacles, engineered to pack opposition voters "
    "into as few districts as possible or spread them thin so they can't win anywhere.",
    ps["body"]))
story.append(Paragraph(
    "This project answers: what if a computer drew the districts, using only "
    "population data and a simple set of rules, with no political input at all?",
    ps["body"]))

story.append(Paragraph("Three Requirements Every District Must Meet",ps["h1"]))
story.append(req_row("1","Equal Population",PDF_BLUE,
    f"Every district must have nearly the same number of people. "
    f"For {STATE_NAME}'s {N_DISTRICTS} districts, that's about {target_pop:,.0f} "
    f"people each. The legal standard requires districts be within 1% of that target."))
story.append(Spacer(1,8))
story.append(req_row("2","Contiguous Territory",PDF_GREEN,
    "Every district must be one connected piece of land. You can't have "
    "a district that's two separate blobs with another district between them."))
story.append(Spacer(1,8))
story.append(req_row("3","Compactness",PDF_ORANGE,
    "Districts should be shaped as compactly as possible — more like circles "
    "than snakes. A compact district represents a real community."))
story.append(Spacer(1,12))

story.append(Paragraph("The Algorithm: Population-Bisecting Splitline",ps["h1"]))
story.append(Paragraph(
    "The core idea: keep cutting the state in half by population until "
    "you have the right number of districts.",ps["body"]))
for i,(stitle,stext) in enumerate([
    ("Find the population center",
     "Find the population-weighted center of the state — marked ● on every "
     "process diagram. This is the anchor point for the first bisecting line."),
    ("Draw the first line",
     f"Draw a line at {FIRST_CUT_ANGLE:.0f}° (like a backslash \\) through the "
     f"population center, adjusted so exactly half the population falls on each side. "
     f"The two sides are called Side A (teal) and Side B (salmon) — not left and "
     f"right, since the line is diagonal."),
    ("Repeat inside each region",
     f"Find each new region's population center (●) and draw a new bisecting line "
     f"through it. The seed angle alternates — {FIRST_CUT_ANGLE:.0f}° (NW-SE) for "
     f"odd-depth cuts, {ALT_CUT_ANGLE:.0f}° (NE-SW) for even-depth cuts — to prevent "
     f"successive cuts from all running the same direction and pinwheeling districts "
     f"around the center. Each seed is a goal-seek starting point: the algorithm sweeps "
     f"up to ±{SEARCH_RADIUS:.0f}° to find the angle that best balances population, so "
     f"the actual cut angle is rarely exactly {FIRST_CUT_ANGLE:.0f}° or {ALT_CUT_ANGLE:.0f}°."),
    ("Continue until done",
     f"Repeat until {N_DISTRICTS} regions exist, each with ~{target_pop:,.0f} people. "
     f"The alternating seed directions spread cuts across the compass, so districts "
     f"radiate outward rather than spiraling."),
]):
    story.append(Paragraph(f"Step {i+1}: {stitle}",ps["h2"]))
    story.append(Paragraph(stext,ps["body"]))

story.append(Paragraph("What Makes This Fair?",ps["h1"]))
for pt,txt in [
    ("No political input","The algorithm uses only where people live — nothing "
     "about party, race, income, or voting history. Same data = same map every time."),
    ("Explainable","Every line passes through the population center (●) of its "
     "region at the angle that best balances population on each side."),
    ("Reproducible","Any person or computer running this code on the same Census "
     "data produces identical districts. It cannot be tweaked after the fact."),
    ("The dot matters","The ● shows exactly where the population center was before "
     "each cut. If you see the dot, you know exactly why the line goes where it does."),
]:
    story.append(Paragraph(f"<b>{pt}.</b> {txt}",ps["bullet"]))
    story.append(Spacer(1,4))

# Tolerance box
tol_d=[[Paragraph("<b>The Tolerance Math</b>",
                  ParagraphStyle("TH",fontName="Helvetica-Bold",
                                 fontSize=10,textColor=PDF_NAVY)),""],[
    Paragraph(
        f"For {N_DISTRICTS} districts the algorithm makes {n_splits} splits across "
        f"{MAX_DEPTH} levels. Each split must stay within {MAX_SPLIT_ERROR*100:.3f}% "
        f"of perfect balance to guarantee all final districts within 1%.",
        ParagraphStyle("TB",fontName="Helvetica",fontSize=9,leading=13)),""],[
    Paragraph("Per-split tolerance  =  1%  ÷  ceil(log₂(N_DISTRICTS))",
              ParagraphStyle("TM",fontName="Courier",fontSize=9,
                             textColor=HC("#333333"))),""]]
tol_t=Table(tol_d,colWidths=[6*inch,0*inch])
tol_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),PDF_LGRAY),("BOX",(0,0),(-1,-1),1,PDF_BLUE),
    ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("SPAN",(0,0),(1,0)),("SPAN",(0,1),(1,1)),("SPAN",(0,2),(1,2)),
]))
story.append(tol_t)

# Results table
story.append(Paragraph("District Results Summary",ps["h1"]))
story.append(Paragraph(
    f"Max deviation: {max_dev:.4f}%  |  Result: {result}  |  "
    f"Avg compactness (Polsby-Popper): {pp.mean():.3f}",ps["body"]))
story.append(Paragraph(
    "<b>Population deviation</b> measures how closely each district matches the "
    "equal-population ideal. "
    f"The 2020 Census counted <b>{total_pop:,} residents</b> in {STATE_NAME}. "
    f"Divided equally among {N_DISTRICTS} congressional districts, the target is "
    f"<b>{target_pop:,.0f} people per district</b>. "
    "Deviation is calculated as: "
    "<i>(district population − target) ÷ target × 100%</i>. "
    "A positive deviation means the district has more people than the target; "
    "negative means fewer. "
    "The constitutional standard for congressional districts requires the maximum "
    "deviation across all districts to be under 1%, meaning no district may hold "
    f"more than {target_pop * 1.01:,.0f} or fewer than {target_pop * 0.99:,.0f} "
    f"residents. {STATE_NAME}'s maximum deviation in this map is {max_dev:.4f}% — "
    f"{'within' if max_dev <= MAX_TOTAL_DEV * 100 else 'outside'} the 1% threshold.",
    ps["body"]))
story.append(Paragraph(
    "<b>Compactness (Polsby-Popper score)</b> measures how close a district's shape "
    "is to a circle. The score is computed as 4π × Area ÷ Perimeter². A perfect circle "
    "scores 1.0; elongated or highly irregular shapes score closer to 0. "
    f"In {STATE_NAME}, scores range from "
    f"{min(r['pp_compactness'] for r in summary_rows):.3f} "
    f"(D{min(summary_rows, key=lambda r: r['pp_compactness'])['district']}, "
    f"{escape_xml(min(summary_rows, key=lambda r: r['pp_compactness'])['nearest_city'])} area) "
    f"to "
    f"{max(r['pp_compactness'] for r in summary_rows):.3f} "
    f"(D{max(summary_rows, key=lambda r: r['pp_compactness'])['district']}, "
    f"{escape_xml(max(summary_rows, key=lambda r: r['pp_compactness'])['nearest_city'])} area). "
    "Because this algorithm draws straight-line splits rather than tracing existing roads or "
    "county lines, districts tend to be more compact than those drawn by hand.",
    ps["body"]))
res_d=[["District","Population","Deviation","Rep. Office Area","Area (km²)","Pop/km²","Compact."]]
for r in summary_rows:
    ok="✓" if r["within_1pct"] else "✗"
    res_d.append([f"D{r['district']}",f"{r['population']:,}",
                  f"{r['deviation_pct']:+.3f}% {ok}",
                  f"{r['nearest_city']} ({r['nearest_city_dist_km']:.0f}km)",
                  f"{r['area_km2']:,.0f}",
                  f"{r['pop_density_per_km2']:.1f}",
                  f"{r['pp_compactness']:.3f}"])
res_t=Table(res_d,colWidths=[0.6*inch,0.95*inch,0.95*inch,2.0*inch,0.75*inch,0.65*inch,0.75*inch])
res_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),PDF_NAVY),("TEXTCOLOR",(0,0),(-1,0),PDF_WHITE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[PDF_WHITE,PDF_LGRAY]),
    ("GRID",(0,0),(-1,-1),0.5,PDF_MGRAY),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),8),
]))
story.append(res_t)

# Maps
story.append(PageBreak())
story.append(Paragraph("District Map",ps["h1"]))
story.append(Paragraph(
    "Colored districts with the proposed representative office city for each district "
    "marked with a star (★). Brightness reflects population density. "
    "Each star is placed inside its district boundary.",
    ps["body"]))
story.append(Paragraph(
    "<b>How the representative city is chosen.</b> "
    "For each district, every city inside the district boundary is scored by: "
    "<i>score = log(city population) × exp(−distance / district radius)</i>, "
    "where <i>distance</i> is from the city to the district's population-weighted centroid "
    "and <i>district radius</i> = √(area / π), the radius of a circle with the same area. "
    "The log-population term ensures the chosen city is large enough to host a "
    "congressional office. The exponential distance term rewards cities near where "
    "the district's people actually live — a city at the exact centroid scores the full "
    "log(population) with no penalty, while a city one district-radius away scores about "
    "37% of that. To outscore a centroid city, a more distant city must be substantially "
    "larger: a city two district-radii away needs to be roughly 7× more populous. "
    "The goal is a city that is both accessible to the whole district and significant "
    "enough to support a working congressional office — not simply the largest city "
    "in the district, and not simply the one closest to the geographic centre. "
    "If no qualified city falls inside a district boundary (rare in dense states, "
    "more common in large rural districts), the nearest outside city is used as a fallback.",
    ps["body"]))
story.append(RLImage(MAP_PNG, width=_map_embed_body[0], height=_map_embed_body[1]))
story.append(Spacer(1,14))
story.append(Paragraph(
    "The map below shows the same districts as solid color fills, with splitline "
    "overlays tracing how each district boundary was derived. District labels show "
    "the district number and population in thousands.",
    ps["body"]))
story.append(RLImage(FILLED_PNG, width=_filled_embed[0], height=_filled_embed[1]))
story.append(Spacer(1,14))

# Process pages
for page in range(n_pages):
    story.append(PageBreak())
    split_ids = [page * 2, page * 2 + 1]
    valid_ids  = [si for si in split_ids if si < n_splits]
    title_part = (f"Split {valid_ids[0]+1} & {valid_ids[1]+1}"
                  if len(valid_ids) == 2
                  else f"Split {valid_ids[0]+1}" if valid_ids
                  else "Final Districts")
    story.append(Paragraph(f"How the Districts Were Built — {title_part}", ps["h1"]))

    # Intro text on the first process page only
    if page == 0:
        story.append(Paragraph(
            "Each diagram is zoomed into the sub-region being bisected. "
            "The inset (bottom-right) shows where that sub-region sits within the full state. "
            "Census blocks are shaded teal (Side A) or salmon (Side B) based on which side "
            "of the bisecting line they fall on. "
            "The ● dot marks the population-weighted center of the sub-region — "
            "the anchor point through which every bisecting line passes.",
            ps["body"]))

    # Per-split explanatory text
    for si in valid_ids:
        s = split_log[si]
        seed_dir = "NW–SE" if s["seed_angle"] == FIRST_CUT_ANGLE else "NE–SW"
        total_pop_here = s["pop_left"] + s["pop_right"]
        landed_note = (
            f"The goal-seek started at {s['seed_angle']:.0f}° ({seed_dir}) "
            f"and landed at {s['angle_deg']:.1f}° after sweeping "
            f"{abs(s['angle_deg'] - s['seed_angle']):.0f}° to find the best balance."
            if abs(s["angle_deg"] - s["seed_angle"]) > 0.9
            else f"The seed angle of {s['seed_angle']:.0f}° ({seed_dir}) "
                 f"was already near-optimal (landed at {s['angle_deg']:.1f}°).")
        story.append(Paragraph(
            f"<b>Split {si + 1}:</b> A sub-region of {total_pop_here:,} people is bisected "
            f"through its population center (●). {landed_note} "
            f"Balance error: {s['balance_err']:.3f}%. "
            f"Side A (teal): {s['pop_left']:,} · Side B (salmon): {s['pop_right']:,}.",
            ps["body"]))

    ppath = process_page_path(page + 1)
    if os.path.exists(ppath):
        story.append(RLImage(ppath, width=6.8*inch, height=3.4*inch))

# Border-swap section
# Derive state-specific city examples from the actual run data
_most_compact   = max(summary_rows, key=lambda r: r["pp_compactness"])
_least_compact  = min(summary_rows, key=lambda r: r["pp_compactness"])
_urban_example  = _most_compact["nearest_city"]
_rural_example  = _least_compact["nearest_city"]

story.append(PageBreak())
story.append(Paragraph("Step 2: Border-Swap Refinement", ps["h1"]))
story.append(Paragraph(
    "The splitlines produce geometrically clean cuts but rarely land exactly on "
    "a census block boundary, which leaves small population imbalances. "
    "A border-swap pass corrects this without using any political information.",
    ps["body"]))
story.append(Paragraph("How the swap works", ps["h2"]))
story.append(Paragraph(
    "After all splits are complete, the algorithm builds an adjacency graph — every "
    "census block knows which other blocks it physically touches. "
    f"It then runs up to {N_SWAP_ROUNDS} rounds. In each round, every block on a "
    "district boundary is considered for reassignment: if moving it to the "
    "neighboring district would reduce the maximum population deviation across all "
    "districts, the move is made. Blocks are only ever moved one at a time, and only "
    "when the move strictly improves balance. The pass stops early once all districts "
    "are within the 1% tolerance.",
    ps["body"]))
story.append(Paragraph("Why the boundaries look jagged", ps["h2"]))
story.append(Paragraph(
    f"The straight splitline is superseded by the swap. The final district boundary "
    f"is the outer edge of whichever census blocks belong to each district after "
    f"swapping — a path that follows census block edges. In {STATE_NAME}, census "
    f"blocks vary widely in size: tiny blocks in dense urban areas like "
    f"{escape_xml(_urban_example)} versus large blocks in sparse rural areas like "
    f"the {escape_xml(_rural_example)} region. The boundary is jagged where blocks "
    f"are small and tightly packed, and smooth where blocks are large and sparse.",
    ps["body"]))
if os.path.exists(SWAP_PNG):
    story.append(Paragraph(
        "The two maps below show the full state before and after the swap step. "
        "Left: districts colored by the splitline assignment alone, with bisection "
        "lines overlaid. Right: the final assignment after swapping — boundaries "
        "now follow census block edges rather than straight geometric lines.",
        ps["body"]))
    _swap_embed = _img_embed(SWAP_PNG, 6.8, max_h_in=5.5)
    story.append(RLImage(SWAP_PNG, width=_swap_embed[0], height=_swap_embed[1]))
if os.path.exists(SWAP_ZOOM_PNG):
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The map below is zoomed into the area with the most block reassignments, "
        "showing the original bisection lines overlaid on the final district colors. "
        "The straight lines mark where the algorithm made its cuts; the jagged color "
        "boundaries show where census blocks actually fall.",
        ps["body"]))
    _zoom_embed = _img_embed(SWAP_ZOOM_PNG, 6.8)
    story.append(RLImage(SWAP_ZOOM_PNG, width=_zoom_embed[0], height=_zoom_embed[1]))

# Appendix
story.append(PageBreak())
story.append(Paragraph("Appendix: Key Algorithm Functions",ps["h1"]))
story.append(Paragraph(
    f"The full source code is in a separate PDF: "
    f"<b>{os.path.basename(CODE_PDF)}</b>. "
    "Below are the three core functions that define the algorithm.",ps["body"]))

with open(script_path,"r") as fh: all_source=fh.read()
key_fns=[
    ("goal_seek_angle",
     "Finds the angle through the population center that best bisects the "
     "population. Sweeps outward from the seed angle, accepts the first angle "
     "within tolerance, falls back to a full 180° sweep if needed."),
    ("splitline",
     "The recursive heart of the algorithm. Bisects a set of census blocks "
     "into equal-population districts. Each call handles one split, then "
     "recurses on both halves."),
    ("balance_at_angle",
     "Divides all blocks by a line through the anchor point and computes the "
     "population balance error. Called hundreds of times per split."),
]
for fn_name,fn_desc in key_fns:
    story.append(Paragraph(f"Function: {fn_name}()",ps["h2"]))
    story.append(Paragraph(fn_desc,ps["body"]))
    pat=rf"(def {fn_name}\(.*?)\n(?=def |\Z)"
    m=re.search(pat,all_source,re.DOTALL)
    if m:
        fn_src=m.group(1)
        lines=fn_src.split("\n")[:40]
        txt="".join(f"{escape_xml(l)}<br/>" for l in lines)
        if len(fn_src.split("\n"))>40:
            txt+=(f"<i>... ({len(fn_src.split(chr(10)))-40} more lines "
                  f"— see {os.path.basename(CODE_PDF)})</i><br/>")
        story.append(Paragraph(txt,ps["mono"]))
    story.append(Spacer(1,8))

story.append(Spacer(1,12))
story.append(HRFlowable(width="100%",thickness=1,color=PDF_MGRAY,spaceAfter=8))
story.append(Paragraph(
    f"{STATE_NAME} Population-Bisecting Splitline Redistricting {VERSION}  |  {APPORTIONMENT_YEAR} U.S. Census  |  "
    f"{N_DISTRICTS} Congressional Districts  |  Prepared by: {AUTHOR}  |  "
    f"Full source: {os.path.basename(CODE_PDF)}",ps["foot"]))

report_doc=SimpleDocTemplate(REPORT_PDF,pagesize=LETTER_SIZE,
    leftMargin=0.9*inch,rightMargin=0.9*inch,
    topMargin=0.85*inch,bottomMargin=0.85*inch)
report_doc.build(story)
log.info(f"  Saved: {REPORT_PDF}")


# ── Done ──────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"Done — {STATE_NAME} {VERSION}")
log.info(f"  Prepared by:     {AUTHOR}")
log.info(f"  Districts:       {N_DISTRICTS}")
log.info(f"  Max deviation:   {max_dev:.4f}% | {result}")
log.info(f"  Avg compactness: {pp.mean():.3f}")
log.info(f"  Output folder:   {OUTPUT_DIR}/")
log.info(f"  Main report:     {REPORT_PDF}")
log.info(f"  Source code PDF: {CODE_PDF}")
log.info("=" * 60)

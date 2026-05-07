"""
Fair Redistricting v12b
=======================
Fixes from v12:
  - pagesize bug fixed: letter aliased as LETTER_SIZE immediately at import,
    never shadowed by local variables
  - Checkpoint saving: splitline and border-swap results saved to .npy files
    so subsequent runs skip straight to map/PDF generation
  - All other v12 features retained

Requirements:
    pip install geopandas matplotlib shapely scipy numpy pandas tqdm
    pip install reportlab pypdf pyproj pillow

To run for any state, change STATE_NAME below, then:
    python redistricting_v12b.py

To regenerate maps/PDF only (reuse saved computation):
    python redistricting_v12b.py  # will auto-detect checkpoint
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
import os, zipfile, urllib.request, logging, math, re
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from shapely.geometry import LineString
from shapely.ops import unary_union
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
STATE_NAME = "Colorado"
AUTHOR     = "Steve Stanzel, Boulder CO"
# ═══════════════════════════════════════════════════════════════

VERSION = "v12b"

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

COLORADO_CITIES = [
    ("Denver",            39.7392, -104.9903),
    ("Colorado Springs",  38.8339, -104.8214),
    ("Aurora",            39.7294, -104.8319),
    ("Fort Collins",      40.5853, -105.0844),
    ("Lakewood",          39.7047, -105.0814),
    ("Thornton",          39.8680, -104.9719),
    ("Arvada",            39.8028, -105.0875),
    ("Westminster",       39.8367, -105.0372),
    ("Pueblo",            38.2544, -104.6091),
    ("Centennial",        39.5807, -104.8772),
    ("Boulder",           40.0150, -105.2705),
    ("Highlands Ranch",   39.5480, -104.9694),
    ("Greeley",           40.4233, -104.7091),
    ("Longmont",          40.1672, -105.1019),
    ("Loveland",          40.3978, -105.0749),
    ("Broomfield",        39.9205, -105.0866),
    ("Castle Rock",       39.3722, -104.8561),
    ("Commerce City",     39.8083, -104.9339),
    ("Parker",            39.5186, -104.7614),
    ("Northglenn",        39.8872, -104.9811),
    ("Brighton",          39.9855, -104.8197),
    ("Pueblo West",       38.3442, -104.7231),
    ("Security-Widefield",38.7483, -104.7147),
    ("Fountain",          38.6822, -104.7008),
    ("Erie",              40.0503, -105.0469),
    ("Frederick",         40.0997, -104.9414),
    ("Windsor",           40.4775, -104.9014),
    ("Evans",             40.3758, -104.6914),
    ("Firestone",         40.1542, -104.9442),
    ("Littleton",         39.6133, -105.0166),
    ("Englewood",         39.6486, -104.9878),
    ("Wheat Ridge",       39.7661, -105.0772),
    ("Golden",            39.7555, -105.2211),
    ("Lone Tree",         39.5272, -104.8725),
    ("Greenwood Village", 39.6197, -104.8911),
    ("Sheridan",          39.6444, -105.0175),
    ("Manitou Springs",   38.8594, -104.9161),
    ("Monument",          39.0928, -104.8728),
    ("Black Forest",      39.0181, -104.6894),
    ("Woodland Park",     38.9939, -105.0569),
    ("Canon City",        38.4408, -105.2428),
    ("Florence",          38.3908, -105.1172),
    ("La Junta",          37.9847, -103.5430),
    ("Trinidad",          37.1694, -104.5003),
    ("Lamar",             38.0872, -102.6207),
    ("Walsenburg",        37.6236, -104.7819),
    ("Alamosa",           37.4697, -105.8700),
    ("Monte Vista",       37.5797, -106.1486),
    ("Del Norte",         37.6769, -106.3544),
    ("Antonito",          37.0814, -106.0103),
    ("Grand Junction",    39.0639, -108.5506),
    ("Montrose",          38.4783, -107.8762),
    ("Durango",           37.2753, -107.8801),
    ("Glenwood Springs",  39.5505, -107.3248),
    ("Steamboat Springs", 40.4850, -106.8317),
    ("Craig",             40.5153, -107.5465),
    ("Cortez",            37.3489, -108.5859),
    ("Telluride",         37.9375, -107.8123),
    ("Delta",             38.7397, -108.0756),
    ("Rifle",             39.5316, -107.7832),
    ("Fruita",            39.1583, -108.7287),
    ("Palisade",          39.1152, -108.3540),
    ("Meeker",            40.0372, -107.9123),
    ("Rangely",           40.0875, -108.8014),
    ("Pagosa Springs",    37.2694, -107.0097),
    ("Silverton",         37.8119, -107.6648),
    ("Aspen",             39.1911, -106.8175),
    ("Vail",              39.6433, -106.3781),
    ("Breckenridge",      39.4817, -106.0384),
    ("Keystone",          39.6061, -105.9678),
    ("Dillon",            39.6294, -106.0442),
    ("Silverthorne",      39.6325, -106.0694),
    ("Frisco",            39.5742, -106.0978),
    ("Edwards",           39.6450, -106.5953),
    ("Eagle",             39.6553, -106.8283),
    ("Avon",              39.6317, -106.5228),
    ("Gypsum",            39.6469, -106.9514),
    ("Leadville",         39.2508, -106.2925),
    ("Buena Vista",       38.8422, -106.1317),
    ("Salida",            38.5347, -105.9986),
    ("Sterling",          40.6255, -103.2077),
    ("Fort Morgan",       40.2502, -103.7999),
    ("Brush",             40.2591, -103.6260),
    ("Yuma",              40.1244, -102.7157),
    ("Wray",              40.0755, -102.2227),
    ("Burlington",        39.3008, -102.2710),
    ("Springfield",       37.4072, -102.6185),
    ("Fort Lupton",       40.0836, -104.8031),
    ("Estes Park",        40.3772, -105.5217),
    ("Granby",            40.0880, -105.9425),
    ("Kremmling",         40.0586, -106.3886),
    ("Walden",            40.7322, -106.2836),
]

# ── Validate ──────────────────────────────────────────────────
if STATE_NAME not in STATES:
    print(f"ERROR: '{STATE_NAME}' not in STATES.")
    sys.exit(1)

STATE       = STATES[STATE_NAME]
STATE_FIPS  = STATE["fips"]
N_DISTRICTS = STATE["districts"]

FIRST_CUT_ANGLE = 135.0
ALT_CUT_ANGLE   = 45.0
MAX_TOTAL_DEV   = 0.01
SEARCH_RADIUS   = 45.0
N_SWAP_ROUNDS   = 200

if N_DISTRICTS == 1:
    MAX_DEPTH = 1; MAX_SPLIT_ERROR = MAX_TOTAL_DEV
else:
    MAX_DEPTH = math.ceil(math.log2(N_DISTRICTS))
    MAX_SPLIT_ERROR = MAX_TOTAL_DEV / MAX_DEPTH

STATE_SLUG = STATE_NAME.replace(" ", "_")
OUTPUT_DIR = f"redistricting_{STATE_SLUG}_{VERSION}"
CACHE_CSV  = f"{STATE_FIPS}_{STATE_SLUG}_centroids_cache.csv"
SHP_DIR    = f"{STATE_FIPS}_blocks"
ZIP_PATH   = f"tl_2020_{STATE_FIPS}_tabblock20.zip"
CHECKPOINT = os.path.join(OUTPUT_DIR, f"checkpoint_{STATE_SLUG}.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output paths — all variables, never hardcoded strings
MAP_PNG      = os.path.join(OUTPUT_DIR, f"districts_{STATE_SLUG}_{VERSION}.png")
SUMMARY_PNG  = os.path.join(OUTPUT_DIR, f"summary_{STATE_SLUG}_{VERSION}.png")
BLOCKS_PNG   = os.path.join(OUTPUT_DIR, f"census_blocks_{STATE_SLUG}_{VERSION}.png")
REPORT_PDF   = os.path.join(OUTPUT_DIR, f"report_{STATE_SLUG}_{VERSION}.pdf")
CODE_PDF     = os.path.join(OUTPUT_DIR, f"source_code_{STATE_SLUG}_{VERSION}.pdf")

def process_page_path(n):
    return os.path.join(OUTPUT_DIR,
        f"process_page{n}_{STATE_SLUG}_{VERSION}.png")

MAP_COLORS = [
    "#4E79A7","#F28E2B","#59A14F","#E15759",
    "#76B7B2","#EDC948","#B07AA1","#FF9DA7",
    "#9C755F","#BAB0AC","#D37295","#A0CBE8",
    "#86BCB6","#F1CE63","#B6992D","#499894",
]
SPLIT_COLORS = {0:"#C0392B",1:"#E67E22",2:"#27AE60",3:"#8E44AD",4:"#2980B9"}
SPLIT_WIDTH  = {0:2.8,1:2.2,2:1.8,3:1.4,4:1.2}

# ── Logging ───────────────────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR,
    f"run_{STATE_SLUG}_{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_path, mode="w")])
log = logging.getLogger()
log.info("=" * 60)
log.info(f"Fair Redistricting {VERSION} — {STATE_NAME}")
log.info(f"  Prepared by: {AUTHOR}")
log.info(f"  Districts: {N_DISTRICTS} | Depth: {MAX_DEPTH} | "
         f"Per-split tol: {MAX_SPLIT_ERROR*100:.3f}%")
log.info("=" * 60)

if N_DISTRICTS == 1:
    log.info(f"{STATE_NAME} has 1 district — no splitting needed.")
    sys.exit(0)


# ── Stage 1: Load ─────────────────────────────────────────────
log.info("[Stage 1] Loading census block centroids")
URL = (f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
       f"tl_2020_{STATE_FIPS}_tabblock20.zip")

if os.path.exists(CACHE_CSV):
    log.info(f"  Cache: {CACHE_CSV}")
    df      = pd.read_csv(CACHE_CSV)
    cx_all  = df["cx"].values;   cy_all  = df["cy"].values
    pop_all = df["POP20"].values; geoids  = df["GEOID20"].values
    log.info(f"  {len(df):,} blocks. Loading geometries...")
    shp = [f for f in os.listdir(SHP_DIR) if f.endswith(".shp")][0]
    gdf = gpd.read_file(os.path.join(SHP_DIR, shp))
    gdf = gdf.to_crs(epsg=26913)
    gdf["POP20"] = gdf["POP20"].astype(int)
    gdf = gdf[(gdf["ALAND20"]>0)&(gdf["POP20"]>0)].copy().reset_index(drop=True)
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
else:
    if not os.path.exists(SHP_DIR):
        log.info(f"  Downloading {STATE_NAME} shapefile...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH,"r") as z: z.extractall(SHP_DIR)
    shp = [f for f in os.listdir(SHP_DIR) if f.endswith(".shp")][0]
    log.info("  Loading shapefile...")
    gdf = gpd.read_file(os.path.join(SHP_DIR, shp))
    gdf = gdf.to_crs(epsg=26913)
    gdf["POP20"] = gdf["POP20"].astype(int)
    gdf = gdf[(gdf["ALAND20"]>0)&(gdf["POP20"]>0)].copy().reset_index(drop=True)
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
    gdf[["GEOID20","POP20","cx","cy"]].to_csv(CACHE_CSV, index=False)
    log.info(f"  Cache saved: {CACHE_CSV}")
    cx_all  = gdf["cx"].values;   cy_all  = gdf["cy"].values
    pop_all = gdf["POP20"].values; geoids  = gdf["GEOID20"].values

total_pop  = pop_all.sum()
target_pop = total_pop / N_DISTRICTS
n_blocks   = len(cx_all)
xmin_s,xmax_s = cx_all.min(),cx_all.max()
ymin_s,ymax_s = cy_all.min(),cy_all.max()
log.info(f"  {n_blocks:,} blocks | pop: {total_pop:,} | target: {target_pop:,.0f}")

utm_to_wgs84 = Transformer.from_crs("EPSG:26913","EPSG:4326",always_xy=True)

def utm_to_latlon(utm_x, utm_y):
    lon, lat = utm_to_wgs84.transform(utm_x, utm_y)
    return round(lat,4), round(lon,4)


# ── Stage 2: Split functions ──────────────────────────────────
def weighted_centroid(indices):
    w = pop_all[indices]
    return (np.average(cx_all[indices],weights=w),
            np.average(cy_all[indices],weights=w))

def balance_at_angle(indices, angle_rad, ax, ay):
    dx,dy = np.cos(angle_rad),np.sin(angle_rad)
    signed = (cx_all[indices]-ax)*(-dy)+(cy_all[indices]-ay)*dx
    lm=signed<=0; rm=~lm
    pl=pop_all[indices][lm].sum(); pr=pop_all[indices][rm].sum()
    total=pl+pr
    err=abs(pl-pr)/total if total>0 else 1.0
    return lm,rm,err,int(pl),int(pr)

def goal_seek_angle(indices, ax, ay, seed_angle_deg):
    """
    Sweep outward from seed_angle_deg in both directions simultaneously.
    Accept first angle within SEARCH_RADIUS achieving balance <= MAX_SPLIT_ERROR.
    Fall back to full 180 sweep if needed.
    """
    best_err=np.inf; best_angle=np.deg2rad(seed_angle_deg)
    best_left=best_right=None; best_pops=(0,0)
    for delta in range(0,int(SEARCH_RADIUS)+1):
        for sign in ([0] if delta==0 else [1,-1]):
            ang=(seed_angle_deg+sign*delta)%180
            ar=np.deg2rad(ang)
            lm,rm,err,pl,pr=balance_at_angle(indices,ar,ax,ay)
            if err<best_err:
                best_err=err; best_angle=ar
                best_left=lm; best_right=rm; best_pops=(pl,pr)
            if err<=MAX_SPLIT_ERROR:
                return best_angle,best_left,best_right,best_err,best_pops
    log.info(f"    [!] Expanding to full 180° (best: {best_err*100:.3f}%)")
    for deg in range(0,180):
        ar=np.deg2rad(deg)
        lm,rm,err,pl,pr=balance_at_angle(indices,ar,ax,ay)
        if err<best_err:
            best_err=err; best_angle=ar
            best_left=lm; best_right=rm; best_pops=(pl,pr)
        if err<=MAX_SPLIT_ERROR: break
    return best_angle,best_left,best_right,best_err,best_pops

def make_clipped_line(ax, ay, angle_rad, region_shape):
    """Extend split line infinitely, clip to sub-region boundary."""
    dx,dy = np.cos(angle_rad),np.sin(angle_rad)
    L=2e6
    full=LineString([(ax-L*dx,ay-L*dy),(ax+L*dx,ay+L*dy)])
    try: return full.intersection(region_shape)
    except Exception: return full

def region_shape_from_indices(indices):
    return unary_union(gdf.iloc[indices].geometry.values)


# ── Stage 3: Splitline (with checkpoint) ─────────────────────
if os.path.exists(CHECKPOINT):
    log.info(f"[Stage 3] Loading checkpoint: {CHECKPOINT}")
    ckpt    = np.load(CHECKPOINT, allow_pickle=True).item()
    labels  = ckpt["labels"]
    # split_log can't be fully reconstructed from checkpoint (geometry objects)
    # so we rebuild a simplified version for visualization
    split_log_data = ckpt["split_log_data"]
    log.info(f"  Loaded {len(split_log_data)} splits from checkpoint.")

    # Rebuild split_log entries without geometry (for PDF text/stats only)
    # Clipped lines will be recomputed for drawing
    log.info("  Rebuilding split geometries for visualization...")
    gdf["district"] = labels

    split_log = []
    for sd in split_log_data:
        indices   = np.where(np.isin(np.arange(n_blocks), sd["indices_list"]))[0]
        left_mask = np.isin(np.arange(len(indices)),
                            [i for i,idx in enumerate(indices)
                             if idx in sd["left_indices_set"]])
        # Recompute clipped line
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
        Alternates between 135 deg (NW-SE) and 45 deg (NE-SW) seed angles.
        """
        n_total = n_left_d+n_right_d
        if n_total==1: labels[indices]=d_start; return
        if len(indices)==0: return

        ax,ay     = weighted_centroid(indices)
        split_num = split_counter[0]
        seed      = FIRST_CUT_ANGLE if split_num%2==0 else ALT_CUT_ANGLE
        split_counter[0] += 1

        angle_rad,left_mask,right_mask,err,(pl,pr) = \
            goal_seek_angle(indices,ax,ay,seed)
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
    # Also save index arrays for reconstruction
    for i,s in enumerate(split_log):
        split_log_data[i]["indices_list"]    = s["indices"].tolist()
        split_log_data[i]["left_indices_set"] = set(
            s["indices"][s["left_mask"]].tolist())

    np.save(CHECKPOINT, {"labels": labels,
                          "split_log_data": split_log_data},
            allow_pickle=True)
    log.info(f"  Checkpoint saved: {CHECKPOINT}")


# ── Stage 4: Border-swap ──────────────────────────────────────
SWAP_CHECKPOINT = os.path.join(OUTPUT_DIR,
    f"checkpoint_swap_{STATE_SLUG}.npy")

if os.path.exists(SWAP_CHECKPOINT):
    log.info(f"[Stage 4] Loading swap checkpoint: {SWAP_CHECKPOINT}")
    labels = np.load(SWAP_CHECKPOINT, allow_pickle=True).item()["labels"]
    gdf["district"] = labels
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
log.info("[Stage 5] Dissolving district shapes...")
district_shapes = gdf.dissolve(by="district")
hull = district_shapes.geometry.convex_hull
pp   = 4*np.pi*hull.area/hull.length**2
log.info(f"  Avg compactness: {pp.mean():.3f}")


# ── Stage 6: City lookup ──────────────────────────────────────
log.info("[Stage 6] Nearest city lookup")

def haversine(lat1,lon1,lat2,lon2):
    R=6371
    phi1,phi2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlam=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(a))

def nearest_city(utm_x, utm_y):
    lat,lon = utm_to_latlon(utm_x,utm_y)
    best_d=float('inf'); best_c="Unknown"
    for city,clat,clon in COLORADO_CITIES:
        d=haversine(lat,lon,clat,clon)
        if d<best_d: best_d=d; best_c=city
    return best_c, round(best_d,1), lat, lon

summary_rows = []
for d in range(N_DISTRICTS):
    idx=np.where(labels==d)[0]; w=pop_all[idx]
    cx=np.average(cx_all[idx],weights=w)
    cy=np.average(cy_all[idx],weights=w)
    city,dist_km,lat,lon = nearest_city(cx,cy)
    ok="✓" if abs(dev_pct[d])<=MAX_TOTAL_DEV*100 else "✗"
    log.info(f"  D{d+1:<3} {pop_stats[d]:>10,} {dev_pct[d]:>+7.3f}%  "
             f"{city:<22} {dist_km:>5.0f}km  {ok}")
    summary_rows.append({
        "district":d+1, "population":int(pop_stats[d]),
        "deviation_pct":round(float(dev_pct[d]),4),
        "within_1pct":abs(dev_pct[d])<=MAX_TOTAL_DEV*100,
        "pp_compactness":round(float(pp.iloc[d]),4),
        "center_lat":lat, "center_lon":lon,
        "nearest_city":city, "nearest_city_dist_km":dist_km,
    })
summary_df = pd.DataFrame(summary_rows)


# ── Drawing helpers ───────────────────────────────────────────
def set_state_bounds(ax, pad=15000):
    ax.set_xlim(xmin_s-pad, xmax_s+pad)
    ax.set_ylim(ymin_s-pad, ymax_s+pad)
    ax.set_axis_off()

def draw_clipped_splits(ax, splits):
    for s in splits:
        col=SPLIT_COLORS.get(s["level"],"#555")
        wid=SPLIT_WIDTH.get(s["level"],1.0)
        line=s["clipped_line"]
        if line is None or line.is_empty: continue
        geoms=[line] if line.geom_type=="LineString" else list(line.geoms)
        for g in geoms:
            if g.geom_type!="LineString": continue
            xs,ys=g.xy
            ax.plot(xs,ys,color=col,linewidth=wid*1.8,
                    linestyle="--",alpha=0.92,zorder=6)
        ax.plot(s["anchor_x"],s["anchor_y"],"o",color=col,markersize=8,
                zorder=7,markeredgecolor="white",markeredgewidth=1.5)

def draw_region_labels(ax, s, fontsize=11):
    for idx_set,pop_val in [
        (s["indices"][s["left_mask"]],  s["pop_left"]),
        (s["indices"][~s["left_mask"]], s["pop_right"]),
    ]:
        if len(idx_set)==0: continue
        w=pop_all[idx_set]
        lx=np.average(cx_all[idx_set],weights=w)
        ly=np.average(cy_all[idx_set],weights=w)
        ax.annotate(f"{pop_val/1000:.0f}k",(lx,ly),
                    ha="center",va="center",fontsize=fontsize,fontweight="500",
                    bbox=dict(boxstyle="round,pad=0.4",facecolor="white",
                              alpha=0.90,edgecolor="#AAAAAA",linewidth=0.5),zorder=9)

def split_legend_handles(include_dot=True):
    levels=sorted(set(s["level"] for s in split_log))
    handles=[Line2D([0],[0],color=SPLIT_COLORS.get(l,"#555"),
                    linewidth=SPLIT_WIDTH.get(l,1)*1.8,linestyle="--",
                    label=f"Level {l} cuts") for l in levels]
    if include_dot:
        handles.append(Line2D([0],[0],marker="o",color="w",
                               markerfacecolor="#555555",markersize=9,
                               markeredgecolor="white",markeredgewidth=1.2,
                               label="● Population center of sub-region\n"
                                     "  (bisection anchor point)"))
    return handles


# ── Stage 7: Census block explainer ───────────────────────────
log.info("[Stage 7] Census block explainer map...")
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

# Inset boxes (UTM coords)
DEN_X0,DEN_X1,DEN_Y0,DEN_Y1 = 488000,528000,4380000,4420000
COS_X0,COS_X1,COS_Y0,COS_Y1 = 508000,536000,4280000,4310000
RUR_X0,RUR_X1,RUR_Y0,RUR_Y1 = 700000,760000,4330000,4390000

fig_b,ax_b = plt.subplots(figsize=(16,10))
fig_b.patch.set_facecolor("#1A1A2E"); ax_b.set_facecolor("#1A1A2E")
pop_vals = np.clip(gdf["POP20"].values,1,None)
log_pop  = np.log10(pop_vals)
norm_b   = mcolors.Normalize(vmin=log_pop.min(),vmax=log_pop.max())
cmap_b   = plt.cm.YlOrRd
ax_b.scatter(gdf["cx"].values,gdf["cy"].values,
             c=log_pop,cmap=cmap_b,norm=norm_b,
             s=0.3,alpha=0.6,linewidths=0,zorder=2)

inset_defs_main = [
    (DEN_X0,DEN_Y0,DEN_X1,DEN_Y1,"Denver Metro","A","#00D4FF"),
    (COS_X0,COS_Y0,COS_X1,COS_Y1,"Colorado Springs","B","#FFD700"),
    (RUR_X0,RUR_Y0,RUR_X1,RUR_Y1,"Eastern Plains","C","#98FF98"),
]
for (x0,y0,x1,y1,lbl,letter,col) in inset_defs_main:
    rect=mpatches.Rectangle((x0,y0),x1-x0,y1-y0,
        linewidth=2,edgecolor=col,facecolor="none",zorder=8)
    ax_b.add_patch(rect)
    ax_b.annotate(f"[{letter}] {lbl}",(x0+(x1-x0)/2,y1+8000),
                  ha="center",va="bottom",fontsize=9,
                  color=col,fontweight="bold",zorder=9)

sm_b=plt.cm.ScalarMappable(cmap=cmap_b,norm=norm_b); sm_b.set_array([])
cbar_b=fig_b.colorbar(sm_b,ax=ax_b,shrink=0.5,pad=0.01,
                       label="Population (log scale)")
cbar_b.set_ticks([1,2,3,4])
cbar_b.set_ticklabels(["10","100","1,000","10,000"])
cbar_b.ax.yaxis.label.set_color("white")
cbar_b.ax.tick_params(colors="white")
ax_b.set_title(
    f"{STATE_NAME} — All {blk_stats['n_blocks']:,} Census Blocks\n"
    "Each dot = one block, colored by population (brighter = more people)",
    fontsize=14,fontweight="500",color="white",pad=10)
set_state_bounds(ax_b,pad=20000)

stats_text=(
    f"Census Block Summary\n{'─'*26}\n"
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
ax_b.text(0.02,0.02,stats_text,transform=ax_b.transAxes,
          fontsize=8,verticalalignment="bottom",fontfamily="monospace",
          color="white",
          bbox=dict(boxstyle="round,pad=0.6",facecolor="#0D1B2A",
                    alpha=0.88,edgecolor="#444"))
plt.tight_layout(rect=[0,0,0.88,1])

blocks_main_tmp = os.path.join(OUTPUT_DIR,"blocks_main_tmp.png")
fig_b.savefig(blocks_main_tmp,dpi=130,bbox_inches="tight",
              facecolor=fig_b.get_facecolor())
plt.close()

fig_b2,axes_b2=plt.subplots(1,3,figsize=(16,5))
fig_b2.patch.set_facecolor("#1A1A2E")
inset_detail=[
    (DEN_X0,DEN_Y0,DEN_X1,DEN_Y1,"[A] Denver Metro","#00D4FF"),
    (COS_X0,COS_Y0,COS_X1,COS_Y1,"[B] Colorado Springs","#FFD700"),
    (RUR_X0,RUR_Y0,RUR_X1,RUR_Y1,"[C] Eastern Plains","#98FF98"),
]
for ax_i,(x0,y0,x1,y1,ititle,icol) in zip(axes_b2,inset_detail):
    ax_i.set_facecolor("#0D1B2A")
    imask=(gdf["cx"]>=x0)&(gdf["cx"]<=x1)&(gdf["cy"]>=y0)&(gdf["cy"]<=y1)
    isub=gdf[imask]
    if len(isub)>0:
        ilp=np.log10(np.clip(isub["POP20"].values,1,None))
        inorm=mcolors.Normalize(vmin=ilp.min(),vmax=max(ilp.max(),2))
        for geom,ip in zip(isub.geometry,ilp):
            try:
                xs2,ys2=geom.exterior.xy
                ax_i.fill(xs2,ys2,color=cmap_b(inorm(ip)),alpha=0.8)
                ax_i.plot(xs2,ys2,color="#333333",linewidth=0.2,alpha=0.5)
            except Exception: pass
        for _,rs in isub.nlargest(3,"POP20").iterrows():
            ax_i.annotate(
                f"Pop: {int(rs['POP20']):,}\n{rs['area_sqkm']:.2f} km²",
                (rs["cx"],rs["cy"]),ha="center",va="center",fontsize=7,
                color="white",fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",facecolor="#000",
                          alpha=0.7,edgecolor="none"),zorder=10)
        ax_i.set_title(
            f"{ititle}\n{len(isub):,} blocks | "
            f"{int(isub['POP20'].sum()):,} people | "
            f"median {np.median(isub['area_sqkm']):.2f} km²",
            fontsize=9,color="white",fontweight="500",pad=6)
    ax_i.set_xlim(x0,x1); ax_i.set_ylim(y0,y1)
    for spine in ax_i.spines.values():
        spine.set_edgecolor(icol); spine.set_linewidth(2)
    ax_i.set_xticks([]); ax_i.set_yticks([])

plt.suptitle(
    "Inset Detail: Urban Density vs Rural Sparsity — same data, different scales",
    fontsize=11,color="white",fontweight="500",y=1.02)
plt.tight_layout()
blocks_insets_tmp = os.path.join(OUTPUT_DIR,"blocks_insets_tmp.png")
fig_b2.savefig(blocks_insets_tmp,dpi=130,bbox_inches="tight",
               facecolor=fig_b2.get_facecolor())
plt.close()

img_main   = PILImage.open(blocks_main_tmp)
img_insets = PILImage.open(blocks_insets_tmp)
w_combined = max(img_main.width,img_insets.width)
combined   = PILImage.new("RGB",(w_combined,img_main.height+img_insets.height),
                           "#1A1A2E")
combined.paste(img_main,(0,0))
combined.paste(img_insets,(0,img_main.height))
combined.save(BLOCKS_PNG)
os.remove(blocks_main_tmp); os.remove(blocks_insets_tmp)
log.info(f"  Saved: {BLOCKS_PNG}")


# ── Stage 8: Final district map ───────────────────────────────
log.info("[Stage 8] Final district map...")
fig,axes=plt.subplots(1,2,figsize=(22,10))
fig.patch.set_facecolor("#F8F8F5")

ax=axes[0]; ax.set_facecolor("#D6EAF8")
for d in range(N_DISTRICTS):
    if d in district_shapes.index:
        district_shapes.loc[[d]].plot(ax=ax,color=MAP_COLORS[d%len(MAP_COLORS)],
                                       linewidth=0,alpha=0.90)
district_shapes.boundary.plot(ax=ax,color="white",linewidth=1.8)
draw_clipped_splits(ax,split_log)

for d in range(N_DISTRICTS):
    if d in district_shapes.index:
        c=district_shapes.loc[d].geometry.centroid
        row=summary_rows[d]
        ax.annotate(
            f"D{d+1}  {pop_stats[d]/1000:.0f}k\n"
            f"{dev_pct[d]:+.2f}%\n★ {row['nearest_city']}",
            (c.x,c.y),ha="center",va="center",fontsize=7,fontweight="500",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                      alpha=0.88,edgecolor="none"),zorder=10)

for d in range(N_DISTRICTS):
    idx=np.where(labels==d)[0]; w=pop_all[idx]
    cx=np.average(cx_all[idx],weights=w); cy=np.average(cy_all[idx],weights=w)
    ax.plot(cx,cy,"*",color=MAP_COLORS[d%len(MAP_COLORS)],markersize=11,
            zorder=11,markeredgecolor="white",markeredgewidth=1)

leg_d=[mpatches.Patch(facecolor=MAP_COLORS[d%len(MAP_COLORS)],
        label=f"D{d+1}: {pop_stats[d]/1000:.0f}k ({dev_pct[d]:+.1f}%)")
       for d in range(N_DISTRICTS)]
ax.legend(handles=leg_d+split_legend_handles(),loc="lower left",
          fontsize=7,ncol=2,framealpha=0.92,edgecolor="none")
ax.set_title(
    f"{STATE_NAME} — {N_DISTRICTS} Districts ({APPORTIONMENT_YEAR} apportionment)\n"
    f"Population-Bisecting Splitline {VERSION}  |  "
    f"★ = Rep. office  |  ● = Population center (bisection anchor)",
    fontsize=12,fontweight="500",pad=12)
set_state_bounds(ax)

ax2=axes[1]; ax2.set_facecolor("#D6EAF8")
max_d2=max(dev_pct.abs().max(),MAX_TOTAL_DEV*100*3,0.5)
cmap2=plt.cm.RdYlGn_r
norm2=mcolors.TwoSlopeNorm(vmin=-max_d2,vcenter=0,vmax=max_d2)
for d in range(N_DISTRICTS):
    if d in district_shapes.index:
        district_shapes.loc[[d]].plot(ax=ax2,
            color=cmap2(norm2(float(dev_pct[d]))),linewidth=0,alpha=0.92)
district_shapes.boundary.plot(ax=ax2,color="white",linewidth=1.8)
for d in range(N_DISTRICTS):
    if d in district_shapes.index:
        c=district_shapes.loc[d].geometry.centroid
        ok="✓" if abs(dev_pct[d])<=MAX_TOTAL_DEV*100 else "✗"
        ax2.annotate(f"D{d+1}: {dev_pct[d]:+.3f}% {ok}",
                     (c.x,c.y),ha="center",va="center",fontsize=8,fontweight="500",
                     bbox=dict(boxstyle="round,pad=0.25",facecolor="white",
                               alpha=0.88,edgecolor="none"),zorder=9)
sm2=plt.cm.ScalarMappable(cmap=cmap2,norm=norm2); sm2.set_array([])
fig.colorbar(sm2,ax=ax2,shrink=0.65,pad=0.02,label="Population deviation (%)")
ax2.set_title(f"Population Deviation | Target: <{MAX_TOTAL_DEV*100:.1f}%\n"
              f"Max: {max_dev:.4f}% | {result}",
              fontsize=12,fontweight="500",pad=12)
set_state_bounds(ax2)
plt.suptitle(
    f"{STATE_NAME} Fair Redistricting {VERSION}  |  "
    f"Pop: {total_pop:,}  |  Target: {target_pop:,.0f}/district  |  "
    f"Max deviation: {max_dev:.4f}%  |  {result}",
    fontsize=11,fontweight="500",y=1.01)
plt.tight_layout()
plt.savefig(MAP_PNG,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close()
log.info(f"  Saved: {MAP_PNG}")


# ── Stage 9: Summary chart ────────────────────────────────────
log.info("[Stage 9] Summary chart...")
fig3,axes3=plt.subplots(1,2,figsize=(18,8))
fig3.patch.set_facecolor("#F8F8F5")
ax3=axes3[0]; ax3.set_facecolor("#F8F8F5")
bars=ax3.barh(
    [f"D{r['district']}: {r['nearest_city']}" for r in summary_rows],
    [r["population"] for r in summary_rows],
    color=[MAP_COLORS[(r["district"]-1)%len(MAP_COLORS)] for r in summary_rows],
    alpha=0.88,edgecolor="white",linewidth=0.8)
ax3.axvline(target_pop,color="#C0392B",linewidth=1.5,linestyle="--",
            label=f"Target: {target_pop:,.0f}")
ax3.axvline(target_pop*(1+MAX_TOTAL_DEV),color="#E67E22",linewidth=1,
            linestyle=":",alpha=0.7,label="+1% limit")
ax3.axvline(target_pop*(1-MAX_TOTAL_DEV),color="#E67E22",linewidth=1,
            linestyle=":",alpha=0.7,label="-1% limit")
for bar,row in zip(bars,summary_rows):
    ok="✓" if row["within_1pct"] else "✗"
    ax3.text(bar.get_width()+5000,bar.get_y()+bar.get_height()/2,
             f"{row['deviation_pct']:+.3f}% {ok}",va="center",fontsize=9)
ax3.set_xlabel("Population",fontsize=11)
ax3.set_title("District Populations by Nearest City",fontsize=12,fontweight="500")
ax3.legend(fontsize=9); ax3.grid(True,axis="x",alpha=0.3)
ax4=axes3[1]; ax4.set_axis_off()
td=[["District","Population","Deviation","Rep. Office Area","Compactness"]]
for r in summary_rows:
    ok="✓" if r["within_1pct"] else "✗"
    td.append([f"D{r['district']}",f"{r['population']:,}",
               f"{r['deviation_pct']:+.3f}% {ok}",
               f"{r['nearest_city']}\n({r['nearest_city_dist_km']:.0f}km)",
               f"{r['pp_compactness']:.3f}"])
t4=ax4.table(cellText=td[1:],colLabels=td[0],cellLoc="center",loc="center",
             colWidths=[0.1,0.18,0.18,0.32,0.14])
t4.auto_set_font_size(False); t4.set_fontsize(9); t4.scale(1,1.8)
for (r4,c4),cell in t4.get_celld().items():
    if r4==0: cell.set_facecolor("#1B3A5C"); cell.set_text_props(color="white",fontweight="bold")
    elif r4%2==0: cell.set_facecolor("#F0F4F8")
    else: cell.set_facecolor("white")
    cell.set_edgecolor("#DDDDDD")
ax4.set_title("District Summary Table",fontsize=12,fontweight="500",pad=12)
plt.suptitle(f"{STATE_NAME} Fair Redistricting {VERSION}  |  "
             f"Max deviation: {max_dev:.4f}% | {result}",
             fontsize=11,fontweight="500",y=1.01)
plt.tight_layout()
plt.savefig(SUMMARY_PNG,dpi=150,bbox_inches="tight",facecolor=fig3.get_facecolor())
plt.close()
log.info(f"  Saved: {SUMMARY_PNG}")


# ── Stage 10: Process pages ───────────────────────────────────
log.info("[Stage 10] Process pages...")
n_splits=len(split_log); n_pages=math.ceil(n_splits/2)

for page in range(n_pages):
    fig2,axes2=plt.subplots(1,2,figsize=(22,11))
    fig2.patch.set_facecolor("#F8F8F5")
    for col in range(2):
        split_idx=page*2+col
        ax=axes2[col]
        if split_idx>=n_splits:
            ax.set_facecolor("#D6EAF8")
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    district_shapes.loc[[d]].plot(ax=ax,
                        color=MAP_COLORS[d%len(MAP_COLORS)],linewidth=0,alpha=0.90)
            district_shapes.boundary.plot(ax=ax,color="white",linewidth=1.8)
            draw_clipped_splits(ax,split_log)
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    c=district_shapes.loc[d].geometry.centroid
                    ax.annotate(f"D{d+1}\n{pop_stats[d]/1000:.0f}k",
                                (c.x,c.y),ha="center",va="center",fontsize=8,
                                fontweight="500",
                                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                                          alpha=0.88,edgecolor="none"),zorder=9)
            ax.set_title(f"Final — {N_DISTRICTS} Districts\n"
                         f"Max deviation: {max_dev:.4f}% | {result}",
                         fontsize=14,fontweight="500",pad=12)
        else:
            s=split_log[split_idx]
            splits_so_far=split_log[:split_idx+1]
            ax.set_facecolor("#D6EAF8")
            for d in range(N_DISTRICTS):
                if d in district_shapes.index:
                    district_shapes.loc[[d]].plot(ax=ax,color="#E8E8E8",
                                                   linewidth=0,alpha=0.5)
            district_shapes.boundary.plot(ax=ax,color="#CCCCCC",
                                          linewidth=0.4,alpha=0.6)
            side_a=s["indices"][s["left_mask"]]
            side_b=s["indices"][~s["left_mask"]]
            ax.scatter(cx_all[side_a],cy_all[side_a],c="#5B9BD5",
                       s=1.0,alpha=0.55,linewidths=0,zorder=2)
            ax.scatter(cx_all[side_b],cy_all[side_b],c="#E8785A",
                       s=1.0,alpha=0.55,linewidths=0,zorder=2)
            draw_clipped_splits(ax,splits_so_far)
            draw_region_labels(ax,s,fontsize=11)
            side_leg=[
                mpatches.Patch(facecolor="#5B9BD5",alpha=0.7,
                               label=f"Side A: {s['pop_left']/1000:.0f}k"),
                mpatches.Patch(facecolor="#E8785A",alpha=0.7,
                               label=f"Side B: {s['pop_right']/1000:.0f}k"),
            ]
            ax.legend(handles=side_leg,loc="upper right",
                      fontsize=8,framealpha=0.9,edgecolor="none")
            ax.set_title(
                f"Split {s['split_num']} of {n_splits}  →  "
                f"{s['split_num']+1} regions\n"
                f"Seed: {s['seed_angle']:.0f}° → landed: {s['angle_deg']:.1f}° | "
                f"Balance error: {s['balance_err']:.4f}% | "
                f"Side A: {s['pop_left']/1000:.0f}k  ·  "
                f"Side B: {s['pop_right']/1000:.0f}k\n"
                f"● = population center of sub-region (bisection anchor)",
                fontsize=11,fontweight="500",pad=10)
        set_state_bounds(ax)
    axes2[0].legend(handles=split_legend_handles(include_dot=True),
                    loc="lower left",fontsize=8,framealpha=0.92,edgecolor="none")
    plt.suptitle(
        f"{STATE_NAME} Redistricting {VERSION} — Process Page {page+1} of {n_pages}  |  "
        f"Teal = Side A · Salmon = Side B of bisecting line  |  "
        f"● = Population center before bisection",
        fontsize=11,fontweight="500",y=1.01)
    plt.tight_layout()
    plt.savefig(process_page_path(page+1),dpi=140,bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.close()
    log.info(f"  Saved: {process_page_path(page+1)}")


# ── Stage 11: CSVs ────────────────────────────────────────────
log.info("[Stage 11] Exporting CSVs...")
gdf[["GEOID20","POP20","district"]].to_csv(
    os.path.join(OUTPUT_DIR,f"block_assignments_{STATE_SLUG}_{VERSION}.csv"),
    index=False)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR,f"district_summary_{STATE_SLUG}_{VERSION}.csv"),
    index=False)
pd.DataFrame([{k:v for k,v in s.items()
               if k not in ("indices","left_mask","clipped_line")}
              for s in split_log]).to_csv(
    os.path.join(OUTPUT_DIR,f"split_log_{STATE_SLUG}_{VERSION}.csv"),
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
    f"{STATE_NAME} Fair Redistricting — Full Source Code", ps["code_title"]))
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
    f"{STATE_NAME} Fair Redistricting {VERSION}  |  Full source code  |  "
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

# Cover
story.append(Spacer(1,0.4*inch))
story.append(Paragraph(f"{STATE_NAME}", ps["title"]))
story.append(Paragraph("Fair Congressional Redistricting", ps["sub"]))
story.append(HRFlowable(width="100%",thickness=2,color=PDF_BLUE,spaceAfter=14))
story.append(RLImage(MAP_PNG,width=6.5*inch,height=2.9*inch))
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
    f"through a block. The map below shows every populated block in the state, "
    f"colored by population. Three inset panels compare urban density against "
    f"rural sparsity at finer detail.",ps["body"]))
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
     ["Block pop — max",f"{blk_stats['pop_max']:,}","Densest single block"],]
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
     f"through it, this time at {ALT_CUT_ANGLE:.0f}° (forward slash /). The angles "
     f"alternate for visual variety."),
    ("Continue until done",
     f"Repeat until {N_DISTRICTS} regions exist, each with ~{target_pop:,.0f} people. "
     f"All lines meet near the state's population center, creating a pinwheel pattern."),
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
res_d=[["District","Population","Deviation","Rep. Office Area","Compact."]]
for r in summary_rows:
    ok="✓" if r["within_1pct"] else "✗"
    res_d.append([f"D{r['district']}",f"{r['population']:,}",
                  f"{r['deviation_pct']:+.3f}% {ok}",
                  f"{r['nearest_city']} ({r['nearest_city_dist_km']:.0f}km)",
                  f"{r['pp_compactness']:.3f}"])
res_t=Table(res_d,colWidths=[0.7*inch,1.1*inch,1.1*inch,2.5*inch,0.85*inch])
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
    "Colored districts with proposed representative office cities (★) and "
    "bisection anchor points (●). Split lines are clipped to their sub-region.",
    ps["body"]))
story.append(RLImage(MAP_PNG,width=6.8*inch,height=3.1*inch))
story.append(Spacer(1,14))
story.append(Paragraph("Population Summary",ps["h1"]))
story.append(Paragraph(
    "District populations vs. target, with proposed representative office city.",
    ps["body"]))
story.append(RLImage(SUMMARY_PNG,width=6.8*inch,height=3.1*inch))

# Process pages
for page in range(n_pages):
    story.append(PageBreak())
    end_n=min(page*2+2,n_splits)
    story.append(Paragraph(
        f"How the Districts Were Built — Split {page*2+1}"
        +(f" & {page*2+2}" if page*2+1<n_splits else ""),ps["h1"]))
    story.append(Paragraph(
        "The full state is shown. Census blocks are shaded teal (Side A) or "
        "salmon (Side B) based on which side of the new bisecting line they fall on — "
        "not left or right, since the line is diagonal. "
        "The ● dot marks the population-weighted center of the sub-region being split, "
        "which is the anchor point through which the bisecting line passes.",
        ps["body"]))
    ppath=process_page_path(page+1)
    if os.path.exists(ppath):
        story.append(RLImage(ppath,width=6.8*inch,height=3.4*inch))

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
    f"{STATE_NAME} Fair Redistricting {VERSION}  |  {APPORTIONMENT_YEAR} U.S. Census  |  "
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

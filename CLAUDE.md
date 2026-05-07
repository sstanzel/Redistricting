# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Generates politically-neutral U.S. congressional district maps using a **population-bisecting splitline algorithm** applied to 2020 Census block data. Given a state name, the script downloads shapefiles, recursively splits the state population in half along straight lines until N districts are formed, then fine-tunes via border-swapping to achieve <1% population deviation across districts.

## How to Run

```bash
# Install dependencies (one time)
pip install geopandas matplotlib shapely scipy numpy pandas tqdm reportlab pypdf pyproj pillow

# Run for the configured state
python redistricting_v12b.py
```

The script auto-detects checkpoint files — if a prior run saved `.npy` checkpoint files in the output directory, computation is skipped and only maps/PDFs are regenerated.

## Configuration

All user-facing settings are at the top of the script (lines 93–96):

```python
STATE_NAME = "Colorado"   # Any key from the STATES dict (lines 99–150)
AUTHOR     = "Your Name"
```

Changing `STATE_NAME` automatically sets `N_DISTRICTS`, `STATE_FIPS`, and all output paths. The `STATES` dict covers all 50 states. The `COLORADO_CITIES` list (lines 154–247) is used for city-lookup labels; other states fall back to a lat/lon nearest-city lookup.

## Output

All outputs go to `redistricting_{STATE}_{VERSION}/`:
- `districts_*.png` — Final district map
- `summary_*.png` — Population balance chart
- `census_blocks_*.png` — Raw census block visualization
- `process_page*.png` — Step-by-step algorithm explainer
- `block_assignments_*.csv` — GEOID20 → district number
- `district_summary_*.csv` — Per-district population + compactness
- `split_log_*.csv` — Each splitline's angle and balance error
- `report_*.pdf` — Comprehensive PDF with algorithm explanation and maps
- `source_code_*.pdf` — Full source for reproducibility
- `checkpoint_*.npy` — Saved computation state (delete to force recomputation)

## Architecture

The codebase is a **single monolithic script** (`redistricting_v12b.py`, ~1,440 lines) organized into sequential numbered stages, each delimited by `# ── Stage N` comments. Older per-state scripts (`colorado_redistricting_v*.py`, `redistricting_v9.py` through `redistricting_v12.py`) are preserved for reference; `v12b` is the current version.

**Data flow:**
1. **Load/cache** — Download TIGER shapefile for state FIPS, extract census blocks, cache centroids to `{FIPS}_{STATE}_centroids_cache.csv`
2. **Splitline** — Recursive population bisection: find centroid → sweep angles → split on line that best halves population → recurse on each half; alternates between `FIRST_CUT_ANGLE=135°` and `ALT_CUT_ANGLE=45°` as seed angles per depth level
3. **Border swap** — `N_SWAP_ROUNDS=200` passes of reassigning boundary blocks to neighboring districts to tighten population balance
4. **Dissolve** — `geopandas` `dissolve()` merges blocks into district polygons
5. **City lookup** — Finds nearest named city to each district centroid
6. **Visualizations** — `matplotlib` renders maps, process diagrams, and summary charts
7. **PDF export** — `reportlab` assembles the report; `pypdf` merges source code PDF

**Key constants** (lines 258–268):
- `MAX_TOTAL_DEV = 0.01` — Target ≤1% population deviation across all districts
- `MAX_SPLIT_ERROR = MAX_TOTAL_DEV / MAX_DEPTH` — Per-level tolerance (depth = ⌈log₂(N)⌉)
- `SEARCH_RADIUS = 45.0` — Degrees swept around seed angle when finding balance split
- `MAP_COLORS` — 16-color palette for district rendering

## Versioning Convention

Each version is a standalone `.py` file. Changes between versions are documented in the module docstring at the top of each file. When creating a new version, copy the current file, bump the `VERSION` constant and filename, and document changes in the docstring.

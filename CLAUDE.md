# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Generates politically-neutral U.S. congressional district maps using a **population-bisecting splitline algorithm** applied to 2020 Census block data. Given a state name, the script downloads shapefiles, recursively splits the state population in half along straight lines until N districts are formed, then fine-tunes via border-swapping to achieve <1% population deviation across districts. No political data is used — the same Census data always produces the identical map.

## How to Run

```bash
# Install dependencies (one time)
pip install geopandas matplotlib shapely scipy numpy pandas tqdm reportlab pypdf pyproj pillow openpyxl

# Quick mode — district map PNG only (fast, uses cached checkpoints)
python redistricting.py

# Full report — all maps, process pages, CSVs, and PDF
python redistricting.py --full

# All 50 states, one bundled district-map PDF (planned — not yet implemented)
python redistricting.py --usa_districts

# All 50 states, full reports (planned — not yet implemented)
python redistricting.py --usa_full
```

The script auto-detects checkpoint files — if a prior run completed, heavy computation is skipped and only maps/PDFs are regenerated.

To run a different state, change `STATE_NAME` at the top of the script, then run the same command. Each state's data downloads once into `data/{FIPS}/` and is reused on subsequent runs.

## Configuration

```python
STATE_NAME = "Colorado"   # Any key from the STATES dict
AUTHOR     = "Steve Stanzel, Boulder CO"
```

Changing `STATE_NAME` automatically sets `N_DISTRICTS`, `STATE_FIPS`, and all output paths. The `STATES` dict covers all 50 states with 2020 apportionment seat counts (valid through 2032).

**CRS:** All geometry uses EPSG:5070 (NAD83 Conus Albers Equal-Area, metres). Applies to all contiguous states.

**City data:** `data/cities.csv` — 2020 Census place populations for all 50 states, built by `build_cities.py`. Auto-rebuilt if missing or stale.

## Directory Structure

```
data/{FIPS}/                         # gitignored — downloaded once per state
  tl_2020_{FIPS}_tabblock20.zip
  blocks/                            # extracted Census shapefile
  centroids_cache.csv                # cached block centroids

data/cities.csv                      # git-tracked — all 50 states, ~31k places
data/states/                         # shared state boundary shapefiles + raster masks

output/redistricting_{STATE}_{VERSION}/   # gitignored
  assets/                            # all PNGs
    districts_*.png                  # city-lights district map
    districts_filled_*.png           # solid-color district map (standalone)
    census_complexity_*.png          # census block outline density map
    census_blocks_*.png              # population density map with insets
    summary_*.png                    # population balance chart
    swap_comparison_*.png            # before/after swap map
    swap_zoom_*.png                  # zoomed swap detail
    process_page*.png                # step-by-step splitline diagrams
  data/                              # .npy and .pkl checkpoint files
    checkpoint_*.npy
    checkpoint_swap_*.npy
    checkpoint_dissolve_*.pkl
  logs/                              # run .log files
  block_assignments_*.csv            # GEOID20 → district number
  district_summary_*.csv             # per-district population, deviation, city, area
  split_log_*.csv                    # each splitline's angle and balance error
  report_*.pdf                       # comprehensive PDF report
  source_code_*.pdf                  # full source for reproducibility
```

## Architecture

Single monolithic script (`redistricting.py`, ~2500 lines) organized into sequential numbered stages delimited by `# ── Stage N` comments. Current version tracked via `VERSION` constant and git tags.

**Data flow:**
1. **Load/cache** — Download TIGER shapefile, extract, cache centroids
2. **Splitline** — Recursive population bisection alternating 135°/45° seed angles
3. **Border swap** — Up to `N_SWAP_ROUNDS` passes reassigning boundary blocks to tighten balance
4. **Dissolve** — `geopandas dissolve()` merges blocks into district polygons
5. **City lookup** — `representative_city()`: scores cities inside district by `log(pop) × exp(-dist/radius)`
6. **Visualizations** — city-lights raster map, census block maps, process pages, summary chart
7. **PDF export** — `reportlab` report + `pypdf` source code appendix

**Key constants:**
- `MAX_TOTAL_DEV = 0.01` — ≤1% population deviation target
- `SEARCH_RADIUS = 45.0` — degrees swept around seed angle
- `MAP_COLORS` — 16-color palette

**Census data URL pattern:**
`https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{FIPS}_tabblock20.zip`

## Known Issues / Backlog

- **Label placement** — Dense metro clusters (e.g. Virginia NoVA) cause callout label overlaps and line crossings. Needs cluster-aware placement rewrite.
- **Legend position** — Population density map legend (Stage 7b) can overlap the state outline on tall states. Needs manual legend placement outside state bounds.
- **--usa_districts / --usa_full** — Planned multi-state batch flags not yet implemented.

## Versioning Convention

Bump `VERSION` constant and git tag for each release. Backward-compat checkpoint search automatically checks the prior version's `data/` folder so existing computation is reused.

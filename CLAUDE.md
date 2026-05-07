# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Generates politically-neutral U.S. congressional district maps using a **population-bisecting splitline algorithm** applied to 2020 Census block data. Given a state name, the script downloads shapefiles, recursively splits the state population in half along straight lines until N districts are formed, then fine-tunes via border-swapping to achieve <1% population deviation across districts. No political data is used — the same Census data always produces the identical map.

A companion script (`election_analysis.py`) overlays historical election results onto the proposed districts to show predicted partisan outcomes.

## How to Run

```bash
# Install dependencies (one time)
pip install geopandas matplotlib shapely scipy numpy pandas tqdm reportlab pypdf pyproj pillow openpyxl

# Run redistricting for the configured state
python redistricting.py

# Run election analysis (requires 3 data files — see below)
python election_analysis.py
```

The redistricting script auto-detects checkpoint `.npy` files — if a prior run completed, heavy computation is skipped and only maps/PDFs are regenerated.

## Configuration

All user-facing settings are at the top of the script (lines 93–96):

```python
STATE_NAME = "Colorado"   # Any key from the STATES dict (lines 99–150)
AUTHOR     = "Steve Stanzel, Boulder CO"
```

Changing `STATE_NAME` automatically sets `N_DISTRICTS`, `STATE_FIPS`, and all output paths. The `STATES` dict covers all 50 states with 2020 apportionment seat counts (valid through 2032).

**CRS note:** All geometry uses EPSG:26913 (Colorado UTM Zone 13N, meters). This is accurate for Colorado but must be updated when running other states — use the appropriate UTM zone for the target state.

**City lists:** `COLORADO_CITIES` (lines 154–247) is the only hardcoded state city list. Other states need their own list added or a geocoding API fallback.

## Output

All outputs go to `redistricting_{STATE}_{VERSION}/`:
- `districts_*.png` — Final district map
- `summary_*.png` — Population balance chart
- `census_blocks_*.png` — Raw census block visualization
- `process_page*.png` — Step-by-step algorithm explainer (one page per 2 splits)
- `block_assignments_*.csv` — GEOID20 → district number
- `district_summary_*.csv` — Per-district population, deviation, city, compactness
- `split_log_*.csv` — Each splitline's angle and balance error
- `report_*.pdf` — Comprehensive PDF with algorithm explanation and maps
- `source_code_*.pdf` — Full source for reproducibility
- `checkpoint_*.npy` — Saved computation state (delete to force recomputation)
- `election_analysis/` — Created by `election_analysis.py` (see below)

**Colorado result (v12b):** 8 districts, max deviation 0.61% (PASS ✓), average Polsby-Popper compactness ~0.686.

## Election Analysis Script

`election_analysis.py` is built but blocked on manual data download — the Colorado SOS site blocks bots. Three files must be downloaded by hand:

| File | Source | Save as |
|------|--------|---------|
| 2024 precinct results | coloradosos.org → "2024 General Election precinct level results (XLSX)" | `2024_precinct_results.xlsx` |
| Voter registration | sos.state.co.us → October 2024 voter registration stats | `2024_voter_registration.xlsx` |
| Precinct boundaries | sos.state.co.us → 2024 precinct shapefile | Unzip to `precinct_boundaries/` |

Once files are present, the script: spatial-joins precincts to proposed districts → aggregates 2024 presidential votes (Trump/Harris) → aggregates voter registration (R/D/Unaffiliated) → predicts winner per district → rates competitiveness (Safe >15pt, Likely >8pt, Lean >3pt, Toss-up ≤3pt) → outputs maps, charts, and a PDF report.

**Known:** The script auto-detects column names from the XLSX files and logs them. The `COLUMN CONFIG` section may need manual updates after first run. Spatial join is slow — checkpointing is a planned improvement (not yet implemented).

## Architecture

The codebase is a **single monolithic script** (`redistricting.py`, ~1,440 lines) organized into sequential numbered stages, each delimited by `# ── Stage N` comments. Older versions are in `archive/`; the current version is tracked via the `VERSION` constant and git tags.

**Data flow:**
1. **Load/cache** — Download TIGER shapefile for state FIPS, extract census blocks, cache centroids to `{FIPS}_{STATE}_centroids_cache.csv`
2. **Splitline** — Recursive population bisection: find centroid → sweep angles → split on line that best halves population → recurse on each half; alternates between `FIRST_CUT_ANGLE=135°` and `ALT_CUT_ANGLE=45°` as seed angles per depth level
3. **Border swap** — `N_SWAP_ROUNDS=200` passes of reassigning boundary blocks to neighboring districts to tighten population balance
4. **Dissolve** — `geopandas` `dissolve()` merges blocks into district polygons
5. **City lookup** — Finds nearest named city to each district centroid using pyproj UTM→WGS84 + haversine distance
6. **Visualizations** — `matplotlib` renders maps, process diagrams, and summary charts
7. **PDF export** — `reportlab` assembles the report; `pypdf` merges source code PDF

**Key constants** (lines 258–268):
- `MAX_TOTAL_DEV = 0.01` — Target ≤1% population deviation across all districts
- `MAX_SPLIT_ERROR = MAX_TOTAL_DEV / MAX_DEPTH` — Per-level tolerance (depth = ⌈log₂(N)⌉)
- `SEARCH_RADIUS = 45.0` — Degrees swept around seed angle when finding balance split
- `MAP_COLORS` — 16-color palette for district rendering

**Census data URL pattern:**
`https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{FIPS}_tabblock20.zip`

## Known Issues / Planned (v13)

- **Election analysis checkpointing** — spatial join is slow, should cache to `.npy` like redistricting does
- **Process page label overlap** — population labels (e.g. "2893k") sometimes overlap the split line anchor dot; needs offset logic
- **Census block explainer** — consider adding a scale bar to the dark background map
- **State city lists** — only Colorado is hardcoded; other states need city lists or a geocoding API fallback
- **UTM zone generalization** — EPSG:26913 is hardcoded throughout; needs to be derived from state when running other states

## Versioning Convention

Each version is a standalone `.py` file. Changes between versions are documented in the module docstring at the top of each file. When creating a new version, copy the current file, bump the `VERSION` constant and filename, and document changes in the docstring.

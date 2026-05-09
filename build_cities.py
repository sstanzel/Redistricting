"""
build_cities.py
===============
Generates cities.csv for all 50 states using 2020 Census data.

Data sources (both authoritative Census 2020):
  - Census API: population for every incorporated place / CDP
  - Census TIGER place shapefiles: coordinates (INTPTLAT20, INTPTLON20)

Cached downloads go to data/{FIPS}/places/ (gitignored).
Output data/cities.csv is git-tracked (gitignore exception).

Usage:
    python build_cities.py

Re-run anytime to refresh. Skips downloads already cached.
"""

import os, json, zipfile, urllib.request
import pandas as pd
import geopandas as gpd

# ── Same STATES dict as redistricting.py ──────────────────────
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

OUTPUT_CSV = os.path.join("data", "cities.csv")

all_rows = []

for state_name, info in STATES.items():
    fips = info["fips"]
    places_dir = os.path.join("data", fips, "places")
    zip_path   = os.path.join(places_dir, f"tl_2020_{fips}_place.zip")
    os.makedirs(places_dir, exist_ok=True)

    # ── Download and extract TIGER place shapefile ─────────────
    if not os.path.exists(zip_path):
        url = f"https://www2.census.gov/geo/tiger/TIGER2020/PLACE/tl_2020_{fips}_place.zip"
        print(f"  Downloading {state_name} place shapefile...")
        urllib.request.urlretrieve(url, zip_path)

    shp_files = [f for f in os.listdir(places_dir) if f.endswith(".shp")]
    if not shp_files:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(places_dir)
        shp_files = [f for f in os.listdir(places_dir) if f.endswith(".shp")]

    shp_path = os.path.join(places_dir, shp_files[0])
    gdf = gpd.read_file(shp_path)[["GEOID", "NAME", "INTPTLAT", "INTPTLON"]]
    gdf = gdf.rename(columns={"NAME": "city"})
    gdf["INTPTLAT"] = gdf["INTPTLAT"].astype(float)
    gdf["INTPTLON"] = gdf["INTPTLON"].astype(float)

    # ── Fetch 2020 population from Census API ──────────────────
    api_url = (
        f"https://api.census.gov/data/2020/dec/pl"
        f"?get=NAME,P1_001N,GEO_ID&for=place:*&in=state:{fips}"
    )
    with urllib.request.urlopen(api_url) as resp:
        data = json.loads(resp.read())

    pop_df = pd.DataFrame(data[1:], columns=data[0])
    pop_df["population"] = pd.to_numeric(pop_df["P1_001N"])
    # GEO_ID format: "1600000US0820000" — last 7 chars = GEOID20
    pop_df["GEOID"] = pop_df["GEO_ID"].str[-7:]

    # ── Join and sort ──────────────────────────────────────────
    merged = pop_df.merge(gdf, on="GEOID", how="inner")
    merged = merged[merged["population"] > 0]
    merged = merged.sort_values("population", ascending=False)

    for _, row in merged.iterrows():
        all_rows.append({
            "state":      state_name,
            "city":       row["city"],
            "lat":        round(row["INTPTLAT"], 4),
            "lon":        round(row["INTPTLON"], 4),
            "population": int(row["population"]),
        })

    print(f"  {state_name}: {len(merged)} places")

# ── Write cities.csv ───────────────────────────────────────────
cities_df = pd.DataFrame(all_rows, columns=["state", "city", "lat", "lon", "population"])
cities_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {len(cities_df)} cities across {len(STATES)} states to {OUTPUT_CSV}")

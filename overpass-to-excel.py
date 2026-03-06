# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 01:02:16 2026

@author: ksath
"""
from cmath import log

import geopandas as gpd
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ================= SETTINGS =================

POWER_GEOJSON = "150+500.geojson"  # Exported from Overpass Turbo
ADMIN = "gadm41_IDN.gpkg" # GADM data with coordinate reference system being longtitude/latitude and the WGS84 datum
OUTPUT_FILE = "allocation_by_province.xlsx"

# # ================= LOAD PROVINCES from GADM =================

provinces = gpd.read_file(ADMIN, layer="ADM_ADM_1")
# EPSG:4326, also known as the WGS84 projection, is a coordinate system used in Google Earth and GSP systems. It represents Earth as a three-dimensional ellipsoid
provinces = provinces.to_crs(epsg=4326)
# filter in Java / filter out the rest Indonesia
provinces = provinces[provinces["NAME_1"].isin(['Banten','Jakarta Raya', 'Jawa Barat','Jawa Tengah','Yogyakarta','Jawa Timur'])]
# reproject spatial data (GeoDataFrames) to the Mollweide projection to buffer the polygons a bit to cover extra land area that some plants are located
provinces = provinces.to_crs("ESRI:54009")
provinces["geometry"] = provinces.buffer(350)
# Return to epsg projection
provinces = provinces.to_crs(epsg=4326)


# ================= LOAD POWER FEATURES =================

power_gdf = gpd.read_file(POWER_GEOJSON)
power_gdf = power_gdf.to_crs(epsg=4326)

# Keep only plants and substations
power_gdf = power_gdf[power_gdf["power"].isin(["plant", "substation"])].copy()

# Some ways/relations may not have geometry (rare), so drop them
power_gdf = power_gdf[~power_gdf.geometry.is_empty]

# ================= SPATIAL JOIN =================

# Allocate each power feature to a province
gdf = gpd.sjoin(
    power_gdf,
    provinces[['NAME_1','geometry']],
    how='left',
    predicate='within'
)
gdf.rename(columns={'NAME_1':'province'}, inplace=True)

# Rename the gadm province names to what we know them
province_names_custom = {
    "Banten":"Banten",
    "Jakarta Raya":"DKI Jakarta",
    "Jawa Barat":"Jawa Barat",
    "Jawa Tengah":"Jawa Tengah",
    "Yogyakarta":"DI Yogyakarta",
    "Jawa Timur":"Jawa Timur"}

gdf['province'] = gdf['province'].map(province_names_custom)

# Check if anything failed to join
missing_province = gdf['province'].isna().sum()
if missing_province > 0:
    print(f"⚠️ {missing_province} power features not assigned to any province")
 
unassigned = gdf[gdf['province'].isna()]

if not unassigned.empty:
    # pick whichever column exists
    name_col = 'name' if 'name' in unassigned.columns else 'NAME_1'
    missing_names = sorted(unassigned[name_col].dropna().unique())
    print("Unassigned features:")
    for n in missing_names:
        print(" -", n)

# ================= SPLIT & EXPORT =================

plants = gdf[gdf["power"] == "plant"]
substations = gdf[gdf["power"] == "substation"]

# Define which tags to be exported to excel because there are too many entries from OSM
PLANT_COLS = [
    "landuse",
    "name",
    "name:en",
    "operator",
    "plant:method",
    "plant:output:electricity",
    "plant:source",
    "power",
    "geometry",
    "province"
]

SUBSTATION_COLS = [
    "name",
    "power",
    "voltage",
    "geometry",
    "province"
]

# Clean out plants without proper information
# First, normalise blank strings → NaN before the notna() check
for col in ["name", "name:en", "plant:source"]:
    if col in plants.columns:
        plants[col] = plants[col].replace("", pd.NA)

# Keep plants that have at least one identifier (name, name:en, or plant:source)
plants_cleaned = plants[
    plants[["name", "name:en", "plant:source"]].notna().any(axis=1)
].copy()

print(
    f"Cleaned {len(plants) - len(plants_cleaned)} plants without proper identifiers"
)

# A function to prevent crashes from inconsistent overpass data (because some tags don't exist everywhere)
def keep_existing_columns(gdf, columns):
    return gdf[[c for c in columns if c in gdf.columns]].copy()

# Export only the specified tags
plants_cleaned = keep_existing_columns(plants_cleaned, PLANT_COLS)
substations = keep_existing_columns(substations, SUBSTATION_COLS)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl", mode='w') as writer:
    substations.to_excel(writer, sheet_name="Substations", index=False)
    plants_cleaned.to_excel(writer, sheet_name="Power Plants", index=False)

print("✅ Export complete:", OUTPUT_FILE)

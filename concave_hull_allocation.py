# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:07:41 2026

@author: B245234
"""

# ============================================================
# CREATE NODAL BOUNDARIES FROM OSM POWER DATA (CONCAVE HULL)
# ============================================================

import logging
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
import alphashape

# ---------------------------------------------------------------------------
# LOGGING — replaces ad-hoc print() calls; gives timestamps and severity
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ===================== FILE PATHS ===========================
POWER_GEOJSON = "150+500.geojson"   # from Overpass Turbo
OUTPUT_GPKG = "convex-concave_hull_distribution.gpkg"
OUTPUT_EXCEL = "convex-concave_hull_distribution.xlsx"

# ===================== CRS DEFINITIONS =====================================
CRS_GEOGRAPHIC = "EPSG:4326"
CRS_PROJECTED  = "EPSG:3857"   # metric CRS — safe for distance & area ops

# ===================== ALPHA SHAPE TUNING ==================================
# Lower alpha  → looser / more convex hull
# Higher alpha → tighter / more concave, but risks splitting into fragments
# For single-node groups (< 3 pts) we always fall back to convex hull.
ALPHA = 0.02


# ===================== NODE DEFINITIONS ====================================
NODE_DEFINITIONS: dict[str, list[str]] = {
    "Banten": [
        "PLTU Suralaya",
        "PLTU Banten Serang",
        "PLTU Jawa 7",
        "GITET CIlegon Baru",
    ],
    "Western Jakarta": [
        "PLTGU Muara Karang",
        "Gardu Induk Duri Kosambi",
        "GISTET Kembangan",
        "GITET Balaraja",
        "GITET Lengkong",
    ],
    "Southern Jakarta": [
        "GITET Gandul",
        "Gardu Induk Depok",
        "GITET Cibinong",
    ],
    "Eastern Jakarta": [
        "PLTGU Muara Tawar",
        "GISTET Tambun II",
        "GITET Bekasi",
        "Gardu Induk Cawang Lama",
        "GIS Kandang Sapi",
    ],
    "Bandung Raya": [
        "GITET Bandung Selatan",
        "GITET Tasikmalaya",
        "GITET Ujungberung",
    ],
    "Mandirancan": [
        "GITET Mandirancan",
        "PLTU Cirebon Unit 1",
    ],
    "Cikarang": [
        "GITET Sukatani",
        "GITET Cibatu",
        "GITET Deltamas",
        "PLTGU Jawa-1",
    ],
    "PLTA": [
        "Gardu Induk Cirata",
        "GITET Saguling",
    ],
    "Pemalang": [
        "Gardu Induk Pemalang",
        "GITET BATANG",
    ],
    "Ungaran":     ["GITET Ungaran"],
    "Tanjung Jati": ["PLTU Tanjung Jati B"],
    "Surabaya": [
        "GITET Ngimbang",
        "GITET Krian",
        "Gardu Induk Surabaya Selatan",
    ],
    "Gresik":  ["PLTGU Gresik"],
    "Grati":   ["GITET Grati"],
    "Paiton":  ["GITET Paiton"],
    "Kediri":  ["GITET Kediri"],
    "Solo": [
        "GITET Pedan",
        "GITET Boyolali",
    ],
    "Cilacap": [
        "GITET Kesugihan",
        "PLTU Jawa Tengah 2",
        "GITET GIS Cilacap",
    ],
}

# ===================== COLUMN WHITELIST FOR EXPORT =========================
PLANT_COLS = [
    "landuse", "name", "name:en", "operator",
    "plant:method", "plant:output:electricity", "plant:source", "power", "geometry", "new_node",]

SUBSTATION_COLS = ["name", "power", "voltage", "geometry", "new_node"]

LINE_COLS = ["name", "power", "voltage", "cables", "frequency", "geometry", "new_node"]


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def keep_existing_columns(gdf: gpd.GeoDataFrame, columns: list[str]) -> gpd.GeoDataFrame:
    """Return a copy of *gdf* with only the columns that actually exist."""
    return gdf[[c for c in columns if c in gdf.columns]].copy()


def force_polygon(geom, fallback_points: gpd.GeoSeries) -> Polygon | MultiPolygon:
    """
    Coerce *geom* (alphashape output) to Polygon / MultiPolygon.

    Robustness improvements over the original:
    • Handles None, empty geometry, and every Shapely geometry type.
    • Filters degenerate polygons (area == 0) that can sneak in from
      GeometryCollection members.
    • Falls back to convex hull of the seed centroids when no valid
      polygon can be extracted — guaranteeing a non-None return value.
    """
    # Safe convex-hull fallback built once
    fallback = unary_union(fallback_points).convex_hull

    if geom is None or geom.is_empty:
        log.debug("force_polygon: empty/None geom → convex hull fallback")
        return fallback

    if isinstance(geom, (Polygon, MultiPolygon)):
        # Still guard against zero-area degenerate polygons
        if geom.area > 0:
            return geom
        log.debug("force_polygon: zero-area polygon → convex hull fallback")
        return fallback

    if isinstance(geom, GeometryCollection):
        polys = [
            g for g in geom.geoms
            if isinstance(g, (Polygon, MultiPolygon)) and g.area > 0
        ]
        if polys:
            merged = unary_union(polys)   # merge touching/overlapping fragments
            return merged
        log.debug("force_polygon: GeometryCollection had no valid polys → fallback")
        return fallback

    # Point, LineString, etc. — not useful as a hull
    log.debug("force_polygon: unsupported geometry type %s → fallback", type(geom).__name__)
    return fallback


def build_hull(node_name: str, group_proj: gpd.GeoDataFrame) -> Polygon | MultiPolygon:
    """
    Build a concave hull (alpha shape) for one node group.

    Accuracy improvements:
    • Uses projected centroids so alphashape works in metres, not degrees.
    • Tries progressively smaller alpha values before giving up, so that
      tightly-clustered nodes (e.g. single-point groups that grow with
      more features over time) still get a sensible shape.
    • Always returns a valid Polygon / MultiPolygon via force_polygon.
    """
    points = group_proj.geometry.centroid

    # Need at least 3 distinct points for a meaningful concave hull
    unique_pts = points.drop_duplicates()
    if len(unique_pts) < 3:
        log.info("Node '%s': < 3 distinct points — using convex hull.", node_name)
        return force_polygon(None, points)

    # Try progressively looser alpha until we get a valid polygon.
    # This guards against alpha being too tight for small / collinear clusters.
    for trial_alpha in [ALPHA, ALPHA * 0.5, ALPHA * 0.1, 0]:
        candidate = alphashape.alphashape(points, alpha=trial_alpha)
        result = force_polygon(candidate, points)
        if isinstance(result, (Polygon, MultiPolygon)) and result.area > 0:
            if trial_alpha != ALPHA:
                log.warning(
                    "Node '%s': alpha=%.4f produced degenerate shape; "
                    "fell back to alpha=%.4f.",
                    node_name, ALPHA, trial_alpha,
                )
            return result

    # Should be unreachable, but belt-and-suspenders
    log.error("Node '%s': all alpha attempts failed — returning convex hull.", node_name)
    return unary_union(points).convex_hull


# ===========================================================================
# STEP 1 — LOAD DATA
# ===========================================================================

log.info("Loading %s …", POWER_GEOJSON)
power_gdf = gpd.read_file(POWER_GEOJSON).to_crs(CRS_GEOGRAPHIC)
log.info("Loaded %d features.", len(power_gdf))

# Guard: ensure 'name' column exists (it must exist for seed matching)
if "name" not in power_gdf.columns:
    log.error("Input GeoJSON has no 'name' column — cannot match seed nodes.")
    sys.exit(1)


# ===========================================================================
# STEP 2 — ASSIGN SEED NODES
# ===========================================================================

power_gdf["seed_node"] = None

# Build a reverse lookup: feature name → node label
name_to_node: dict[str, str] = {
    feat_name: node
    for node, names in NODE_DEFINITIONS.items()
    for feat_name in names
}

# Vectorised assignment is cleaner and faster than looping over NODE_DEFINITIONS
power_gdf["seed_node"] = power_gdf["name"].map(name_to_node)

seed_units = power_gdf.dropna(subset=["seed_node"]).copy()

# Warn about any defined seed names that weren't found in the data
found_names = set(power_gdf["name"].dropna())
missing_seeds = {
    node: [n for n in names if n not in found_names]
    for node, names in NODE_DEFINITIONS.items()
    if any(n not in found_names for n in names)
}
if missing_seeds:
    for node, names in missing_seeds.items():
        log.warning("Node '%s': seed name(s) not found in OSM data → %s", node, names)

log.info("Seed units matched: %d / %d defined names.", len(seed_units), len(name_to_node))


# ===========================================================================
# STEP 3 — BUILD CONCAVE HULLS
# ===========================================================================

records = []

for node_name, group in seed_units.groupby("seed_node"):
    group_proj = group.to_crs(CRS_PROJECTED)
    hull_proj  = build_hull(node_name, group_proj)

    # Reproject hull back to geographic CRS for storage & join
    hull_geo = (
        gpd.GeoSeries([hull_proj], crs=CRS_PROJECTED)
        .to_crs(CRS_GEOGRAPHIC)
        .iloc[0]
    )

    records.append({
        "node":       node_name,
        "seed_count": len(group),
        "geometry":   hull_geo,
    })

nodes_gdf = gpd.GeoDataFrame(records, crs=CRS_GEOGRAPHIC)
log.info("Built %d node hulls.", len(nodes_gdf))


# ===========================================================================
# STEP 4 — SPATIAL ALLOCATION (within hull)
# ===========================================================================

# Build a centroid GeoDataFrame in the projected CRS for accurate point-in-
# polygon testing. We only carry the index so we can join the result back.
# This avoids a plant straddling two hulls being double-counted, and fixes
# the CRS warning about centroid on geographic coordinates.
power_proj = power_gdf.to_crs(CRS_PROJECTED)
centroids_gdf = gpd.GeoDataFrame(
    {"orig_index": power_proj.index},
    geometry=power_proj.geometry.centroid,
    crs=CRS_PROJECTED,
)

nodes_proj_join = nodes_gdf[["node", "geometry"]].to_crs(CRS_PROJECTED)

centroid_join = gpd.sjoin(
    centroids_gdf,
    nodes_proj_join,
    how="left",
    predicate="within",
).drop(columns="index_right", errors="ignore")

# sjoin can produce duplicate rows when a centroid falls inside multiple
# overlapping hulls — keep only the first match per original feature.
centroid_join = centroid_join[~centroid_join.index.duplicated(keep="first")]

# Attach the matched node label back onto the original (geographic) GeoDataFrame
power_gdf["node"] = centroid_join["node"]

# Prefer explicit seed assignment over spatial-join result
power_gdf["new_node"] = power_gdf["seed_node"].combine_first(power_gdf["node"])

# 'joined' alias keeps the rest of the script unchanged
joined = power_gdf.drop(columns=["node"], errors="ignore")


# ===========================================================================
# STEP 5 — NEAREST-NODE FALLBACK for unassigned features
# ===========================================================================

unassigned = joined[joined["new_node"].isna()].copy()
assigned   = joined[joined["new_node"].notna()].copy()

if not unassigned.empty:
    log.info("%d features unassigned after spatial join — running nearest-node fallback.", len(unassigned))

    nodes_proj     = nodes_gdf.to_crs(CRS_PROJECTED)
    unassigned_proj = unassigned.to_crs(CRS_PROJECTED)

    # sjoin_nearest matches each unassigned feature to its closest node hull
    nearest = gpd.sjoin_nearest(
        unassigned_proj,
        nodes_proj[["node", "geometry"]],
        how="left",
        distance_col="distance_to_node_m",
        lsuffix="feat",
        rsuffix="node",
    ).drop(columns=["index_right"], errors="ignore")

    nearest["new_node"] = nearest["node"]
    nearest = nearest.drop(columns=["node"], errors="ignore")

    # Reproject back before concat
    nearest = nearest.to_crs(assigned.crs)

    log.info(
        "Nearest-node fallback — max distance: %.2f km, mean: %.2f km.",
        nearest["distance_to_node_m"].max() / 1000,
        nearest["distance_to_node_m"].mean() / 1000,
    )

    # Flag features that ended up very far (> 100 km) — worth reviewing
    far_threshold_m = 100_000
    far_features = nearest[nearest["distance_to_node_m"] > far_threshold_m]
    if not far_features.empty:
        log.warning(
            "%d feature(s) assigned to nearest node but are > 100 km away "
            "— check these manually:\n%s",
            len(far_features),
            far_features[["name", "new_node", "distance_to_node_m"]].to_string(),
        )

    allocated = pd.concat([assigned, nearest], ignore_index=True)
else:
    log.info("All features assigned spatially — no fallback needed.")
    allocated = assigned.copy()


# ===========================================================================
# STEP 6 — FILTER POWER PLANTS FOR GPKG EXPORT
# ===========================================================================

plants_all = allocated[allocated["power"] == "plant"].copy()

# Normalise blank strings → NaN before the notna() check
for col in ["name", "name:en", "plant:source"]:
    if col in plants_all.columns:
        plants_all[col] = plants_all[col].replace("", pd.NA)

# Keep plants that have at least one identifier
plants_gpkg = plants_all[
    plants_all[["name", "name:en", "plant:source"]].notna().any(axis=1)
].copy()

log.info(
    "Plants — total: %d | kept for GPKG: %d | dropped: %d",
    len(plants_all),
    len(plants_gpkg),
    len(plants_all) - len(plants_gpkg),
)


# ===========================================================================
# STEP 7 — EXPORT TO GEOPACKAGE
# ===========================================================================

# Separate feature types from the allocated GeoDataFrame
nodes_out       = nodes_gdf  # the hull polygons — already separate
lines_out       = allocated[allocated["power"] == "line"].copy()
plants_out      = allocated[allocated["power"] == "plant"].copy()
substations_out = allocated[allocated["power"] == "substation"].copy()

# Apply the same identifier filter to plants (same logic as plants_gpkg above)
for col in ["name", "name:en", "plant:source"]:
    if col in plants_out.columns:
        plants_out[col] = plants_out[col].replace("", pd.NA)

plants_out = plants_out[
    plants_out[["name", "name:en", "plant:source"]].notna().any(axis=1)
].copy()

log.info("Writing GPKG → %s", OUTPUT_GPKG)
nodes_out.to_file(OUTPUT_GPKG,       layer="nodes",       driver="GPKG")
lines_out.to_file(OUTPUT_GPKG,       layer="lines",       driver="GPKG")
plants_out.to_file(OUTPUT_GPKG,      layer="power_plants",driver="GPKG")
substations_out.to_file(OUTPUT_GPKG, layer="substations", driver="GPKG")

log.info(
    "GPKG layers — nodes: %d | lines: %d | plants: %d | substations: %d",
    len(nodes_out), len(lines_out), len(plants_out), len(substations_out),
)

# ===========================================================================
# STEP 8 — EXPORT TO EXCEL
# ===========================================================================

# Node centroids (projected → geographic for readable lat/lon)
nodes_proj_export = nodes_gdf.to_crs(CRS_PROJECTED)
centroids_geo = (
    gpd.GeoSeries(nodes_proj_export.geometry.centroid, crs=CRS_PROJECTED)
    .to_crs(CRS_GEOGRAPHIC)
)
nodes_excel = nodes_gdf.copy()
nodes_excel["centroid_lon"] = centroids_geo.x
nodes_excel["centroid_lat"] = centroids_geo.y
nodes_excel = nodes_excel.drop(columns="geometry")

# Split by feature type
plants_excel     = allocated[allocated["power"] == "plant"].copy()
substations_excel = allocated[allocated["power"] == "substation"].copy()
lines_excel       = allocated[allocated["power"] == "line"].copy()

# Filter plants — same logic as GPKG export (remove entries with no identifiers)
for col in ["name", "name:en", "plant:source"]:
    if col in plants_excel.columns:
        plants_excel[col] = plants_excel[col].replace("", pd.NA)

plants_excel = plants_excel[
    plants_excel[["name", "name:en", "plant:source"]].notna().any(axis=1)
].copy()

# Keep only whitelisted columns
plants_excel      = keep_existing_columns(plants_excel,      PLANT_COLS)
substations_excel = keep_existing_columns(substations_excel, SUBSTATION_COLS)
lines_excel       = keep_existing_columns(lines_excel,       LINE_COLS)

# Drop geometry from Excel sheets (not renderable in xlsx)
# plants_excel      = plants_excel.drop(columns="geometry", errors="ignore")
# substations_excel = substations_excel.drop(columns="geometry", errors="ignore")
# lines_excel       = lines_excel.drop(columns="geometry", errors="ignore")

# Sort for readability
plants_excel      = plants_excel.sort_values("new_node")
substations_excel = substations_excel.sort_values("new_node")
lines_excel       = lines_excel.sort_values("new_node")

log.info("Writing Excel → %s", OUTPUT_EXCEL)
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl", mode='w') as writer:
    nodes_excel.to_excel(writer,       sheet_name="Nodes",        index=False)
    plants_excel.to_excel(writer,      sheet_name="Power Plants", index=False)
    substations_excel.to_excel(writer, sheet_name="Substations",  index=False)
    lines_excel.to_excel(writer,       sheet_name="Lines",        index=False)

log.info(
    "Done. Nodes: %d | Plants: %d | Substations: %d | Lines: %d",
    len(nodes_gdf),
    len(plants_excel),
    len(substations_excel),
    len(lines_excel),
)



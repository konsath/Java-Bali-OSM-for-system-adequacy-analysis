import os
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration of population density raster
population_raster_input = "idn_pop_2025_CN_100m_R2025A.tif"
population_raster_output = "javabali_pop_clipped.tif"

# load data
raw_pp_substations = gpd.read_file("150+500.geojson")
jamali = gpd.read_file("javabali_only.geojson")

NODE_DEFINITIONS = {
    "Banten": [
        "PLTU Suralaya",
        # "PLTU Jawa 9 & 10",  # doesn't exist in OSM
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
    "Ungaran": ["GITET Ungaran"],
    "Tanjung Jati": ["PLTU Tanjung Jati B"],
    "Surabaya": [
        "GITET Ngimbang",
        "GITET Krian",
        "Gardu Induk Surabaya Selatan",
    ],
    "Gresik": ["PLTGU Gresik"],
    "Grati": ["GITET Grati"],
    "Paiton": ["GITET Paiton"],
    "Kediri": ["GITET Kediri"],
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

# Extract all facility names from NODE_DEFINITIONS
all_facility_names = [
    facility_name
    for facility_list in NODE_DEFINITIONS.values()
    for facility_name in facility_list
]

# Filter to only keep rows where name is in NODE_DEFINITIONS
gdf = raw_pp_substations[raw_pp_substations["name"].isin(all_facility_names)].copy()
gdf = gdf.reset_index(drop=True)

# save
gdf.to_file("18nodes_VC.gpkg", driver="GPKG", layer="voronoi_points", mode='w')

# Build reverse lookup: facility name -> node name
name_to_node = {
    facility_name: node_name
    for node_name, facility_names in NODE_DEFINITIONS.items()
    for facility_name in facility_names
}

# Assign node_name only when gdf['name'] appears in NODE_DEFINITIONS values
gdf["node_name"] = gdf["name"].map(name_to_node)


# Step 1: generate Voronoi polygons around points
voronoi_geoms = gdf.voronoi_polygons()

# Step 2: create GeoDataFrame for Voronoi polygons
voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_geoms, crs=gdf.crs)

# Step 3: assign attributes by spatial join
# Use the original points as "centroids" to match polygons
voronoi_cells = gpd.sjoin(voronoi_gdf, gdf, how="left", predicate="contains")

# save
voronoi_cells.to_file("18nodes_VC.gpkg", driver="GPKG", layer="voronois", mode='w')

# Clip voronoi cells to jamali boundary
jamali_boundary = jamali.dissolve() # Create a single outer boundary from jamali
voronoi_clipped = voronoi_cells.overlay(jamali_boundary, how='intersection')

# Keep only columns from voronoi_cells (drop any columns added from jamali)
original_cols = list(voronoi_cells.columns)
voronoi_clipped = voronoi_clipped[original_cols]
voronoi_clipped = voronoi_clipped.reset_index(drop=True)

# Drop problematic ID columns before saving
voronoi_clipped = voronoi_clipped.drop(columns=["fid", "id", "@id"], errors="ignore")

# Append as a new layer to the existing GeoPackage
voronoi_clipped.to_file("18nodes_VC.gpkg", driver="GPKG", layer="voronois_clipped", mode='w')

# Merge/dissolve voronoi cells by node_name to create aggregated regions
# Using aggfunc='first' to keep the first value of each attribute per node
nodes_merged = voronoi_clipped.dissolve(by='node_name', aggfunc='first')

print(f"Merged into {len(nodes_merged)} nodes")
nodes_merged.to_file("18nodes_VC.gpkg", driver="GPKG", layer="18nodes", mode='w')

# Clip population density raster to jamali boundary
with rasterio.open(population_raster_input) as src:
    # Reproject jamali boundary to match raster CRS
    jamali_boundary_reproj = jamali_boundary.to_crs(src.crs)
    print(f"\n✓ Reprojected boundary from {jamali.crs} → {src.crs}")
    
    # Convert GeoDataFrame to GeoJSON-like format for rasterio.mask
    # This is the format rasterio expects for the mask geometry
    shapes = [geom for geom in jamali_boundary_reproj.geometry]
    out_image, out_transform = mask(
        src, 
        shapes, 
        crop=True,
        all_touched=True,  # True for more inclusive boundary inclusion at edges, False for simpler cropping
        nodata=src.nodata
        )
    # Update metadata for the output raster
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw"  # Compress output to save disk space (reduces file size by 50-70% without losing data quality)
        # if you want even better compression and don't mind slightly slower read/write, you can use "deflate" instead of "lzw":
        #"compress": "deflate",  # Better compression than LZW
        #"predictor": 2          # Optimizes for population data
    })
    # Write the clipped raster
    with rasterio.open(population_raster_output, "w", **out_meta) as dest:
        dest.write(out_image)
    print(f"✓ Clipped raster saved: {population_raster_output}")    
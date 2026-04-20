"""
autonomous_shrub_agent.py
========================
An autonomous agent that downloads required data layers and generates a shrub prediction
for any given Area of Interest (AOI) provided via a GeoJSON file.

Uses the V12 XGBoost model trained on SAM polygon annotations.
Required inputs: NAIP (4-band) + CHM (canopy height). No DEM needed.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds

# Add parent directory to path so we can import the downloaders and model
sys.path.append(str(Path(__file__).parent.parent))

# Import modular downloaders
import download_naip_all_sites as naip_tool
import download_canopy_height_all_sites as chm_tool
import predict_raster_v12 as prediction_tool

def run_pipeline(geojson_path, output_dir="Model_Prediction", year="2022"):
    """
    Main agentic loop: Detects missing data, downloads it, and runs V12 inference.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    naip_file  = out_path / "naip.tif"
    chm_file   = out_path / "chm.tif"
    shrub_file = out_path / "predicted_shrub_v12.tif"

    print(f"\n[AGENT] Starting Autonomous Shrub Pipeline (V12) for: {geojson_path}")

    # 0. Initialize GEE
    import ee
    print("[AGENT] Initializing Google Earth Engine...")
    try:
        ee.Initialize(project="ee-sefakarabas")
    except Exception:
        print("[AGENT] GEE initialization failed. Attempting authentication...")
        ee.Authenticate()
        ee.Initialize(project="ee-sefakarabas")

    # 1. Load GeoJSON and identify AOI
    gdf = gpd.read_file(geojson_path)
    mask_crs = gdf.crs
    if mask_crs is None:
        print("[AGENT] GeoJSON has no CRS defined. Defaulting to EPSG:4326")
        mask_crs = "EPSG:4326"

    if gdf.crs is None or gdf.crs.is_geographic:
        print("[AGENT] Geographic CRS detected. Projecting AOI to EPSG:6350 (Conus Albers).")
        gdf = gdf.to_crs("EPSG:6350")
        mask_crs = gdf.crs

    bounds     = gdf.total_bounds  # [minx, miny, maxx, maxy]
    bbox_wgs84 = transform_bounds(mask_crs, "EPSG:4326", *bounds)
    print(f"[AGENT] AOI BBox (WGS84): {bbox_wgs84}")

    # 2. NAIP Download
    if not naip_file.exists():
        print(f"[AGENT] NAIP missing. Fetching {year} imagery from Planetary Computer...")
        naip_tool.download_naip_aoi(
            bbox_wgs84=bbox_wgs84,
            mask_crs=mask_crs,
            target_res=1.0,
            year=year,
            output_path=str(naip_file)
        )
    else:
        print(f"[AGENT] NAIP found at {naip_file}")

    # Read NAIP grid info for CHM alignment
    with rasterio.open(naip_file) as src:
        target_width     = src.width
        target_height    = src.height
        target_transform = src.transform
        target_crs       = src.crs

    # 3. CHM Download (Meta Canopy Height) — V12 needs NAIP + CHM only, no DEM
    if not chm_file.exists():
        print("[AGENT] CHM missing. Fetching Meta Canopy Height from Google Earth Engine...")
        chm_tool.download_canopy_aoi(
            bbox_wgs84=bbox_wgs84,
            target_width=target_width,
            target_height=target_height,
            dst_crs=target_crs,
            dst_transform=target_transform,
            output_path=str(chm_file)
        )
    else:
        print(f"[AGENT] CHM found at {chm_file}")

    # 4. Run V12 Shrub Prediction
    print("[AGENT] All base layers ready. Running V12 XGBoost shrub inference...")
    prediction_tool.run_shrub_prediction_v12(
        naip_path=str(naip_file),
        chm_path=str(chm_file),
        output_path=str(shrub_file),
    )

    print(f"\n[AGENT] SUCCESS! Final map saved to: {shrub_file}")
    print(f"        Probability raster: {str(shrub_file).replace('.tif', '_prob.tif')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Shrub Agent (V12 XGBoost)")
    parser.add_argument("geojson", help="Path to GeoJSON file for AOI")
    parser.add_argument("--output", default="Model_Prediction", help="Directory for rasters")
    parser.add_argument("--year",   default="2022", help="NAIP year to fetch")

    args = parser.parse_args()
    run_pipeline(args.geojson, args.output, args.year)

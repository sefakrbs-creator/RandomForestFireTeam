"""
download_canopy_height_all_sites.py
===================================
Generic CHM (Canopy Height Model) downloader for AOI.
Uses Meta Canopy Height dataset from Google Earth Engine.
"""

import ee
import numpy as np
import rasterio
from pathlib import Path
import requests
import tempfile
import os

def download_canopy_aoi(bbox_wgs84, target_width, target_height, dst_crs, dst_transform, output_path):
    """
    Downloads Meta Canopy Height from GEE aligned to a target grid.
    """
    print(f"  Requesting Meta Canopy Height from GEE...")
    
    # Define AOI as ee.Geometry
    region = ee.Geometry.BBox(*bbox_wgs84)
    
    # Load Meta Canopy Height (1m Global)
    chm = ee.Image("projects/meta-forest-monitoring/assets/canopy_height")
    
    # Clip and select the only band
    chm_clipped = chm.clip(region).select('height')
    
    # Prepare download URL
    # We want to match the target grid exactly
    crs_code = str(dst_crs)
    if "EPSG:" not in crs_code:
        # If it's a CRS object, try to get the string
        try:
            crs_code = dst_crs.to_string()
        except:
            pass

    # GEE getDownloadURL
    url = chm_clipped.getDownloadURL({
        'name': 'chm_download',
        'scale': 1.0,
        'crs': crs_code,
        'transform': [dst_transform.a, dst_transform.b, dst_transform.c, 
                      dst_transform.d, dst_transform.e, dst_transform.f],
        'dimensions': [target_width, target_height],
        'format': 'GEO_TIFF'
    })
    
    print(f"  Downloading from GEE URL...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"GEE Download failed: {r.text}")
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"  CHM saved to: {output_path}")

if __name__ == "__main__":
    # Test
    import ee
    ee.Initialize(project="ee-sefakarabas")
    bbox = [-120.45, 37.10, -120.44, 37.11]
    # Dummy transform for 100x100
    from affine import Affine
    aff = Affine(1.0, 0, -120.45, 0, -1.0, 37.11)
    download_canopy_aoi(bbox, 100, 100, "EPSG:4326", aff, "test_chm.tif")

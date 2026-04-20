"""
download_naip_all_sites.py
==========================
Generic NAIP downloader for Area of Interest (AOI).
Uses Planetary Computer STAC API.
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from pathlib import Path
import tempfile
import os

import planetary_computer
from pystac_client import Client

def download_naip_aoi(bbox_wgs84, mask_crs, target_res=1.0, year="2022", output_path="naip.tif"):
    """
    Downloads NAIP imagery for a given WGS84 bounding box.
    """
    print(f"  Searching Planetary Computer for NAIP {year}...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["naip"],
        bbox=list(bbox_wgs84),
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    
    if not items:
        print(f"  No items found for {year}. Trying 2020...")
        search = catalog.search(
            collections=["naip"],
            bbox=list(bbox_wgs84),
            datetime="2020-01-01/2020-12-31",
        )
        items = list(search.items())

    if not items:
        raise RuntimeError("No NAIP tiles found for the given AOI.")

    print(f"  Found {len(items)} tiles. Downloading and cropping...")
    
    tmp_files = []
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84

    for item in items:
        href = item.assets["image"].href
        with rasterio.open(href) as src:
            src_bbox = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
            window = from_bounds(*src_bbox, src.transform)
            # Clip window to source dimensions
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            
            if window.width <= 0 or window.height <= 0:
                continue

            data = src.read(window=window)
            win_transform = src.window_transform(window)

            fd, path = tempfile.mkstemp(suffix=".tif")
            os.close(fd)
            tmp_files.append(path)
            
            with rasterio.open(
                path, "w", driver="GTiff",
                height=data.shape[1], width=data.shape[2], count=src.count,
                dtype=data.dtype, crs=src.crs, transform=win_transform,
            ) as dst:
                dst.write(data)

    if not tmp_files:
        raise RuntimeError("Overlap check failed for all tiles.")

    print(f"  Merging {len(tmp_files)} tiles and reprojecting to {mask_crs}...")
    src_datasets = [rasterio.open(f) for f in tmp_files]
    merged, merged_transform = merge(src_datasets)
    merged_crs = src_datasets[0].crs
    for ds in src_datasets:
        ds.close()

    dst_transform, dst_width, dst_height = calculate_default_transform(
        merged_crs, mask_crs,
        merged.shape[2], merged.shape[1],
        left=merged_transform.c,
        bottom=merged_transform.f + merged_transform.e * merged.shape[1],
        right=merged_transform.c + merged_transform.a * merged.shape[2],
        top=merged_transform.f,
        resolution=target_res,
    )

    dst_data = np.zeros((merged.shape[0], dst_height, dst_width), dtype=merged.dtype)

    for band in range(merged.shape[0]):
        reproject(
            source=merged[band],
            destination=dst_data[band],
            src_transform=merged_transform,
            src_crs=merged_crs,
            dst_transform=dst_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
        )

    # Save to output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=dst_height, width=dst_width,
        count=merged.shape[0], dtype=merged.dtype,
        crs=mask_crs, transform=dst_transform,
        compress="lzw",
    ) as dst:
        dst.write(dst_data)
        for i, name in enumerate(["Red", "Green", "Blue", "NIR"], 1):
            if i <= merged.shape[0]:
                dst.set_band_description(i, name)

    # Cleanup
    for f in tmp_files:
        try: os.unlink(f)
        except: pass

    print(f"  NAIP saved to: {output_path}")

if __name__ == "__main__":
    # Test
    bbox = [-120.45, 37.10, -120.44, 37.11]
    download_naip_aoi(bbox, "EPSG:6350")

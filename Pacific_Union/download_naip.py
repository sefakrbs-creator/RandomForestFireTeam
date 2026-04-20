"""
Download NAIP 4-band imagery (RGBIR) for Pacific Union site via
Microsoft Planetary Computer, cropped to the combined extent of
mask_outputs multiband TIFs, resampled to 1m resolution.
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

# ---------- CONFIG ----------
SITE_DIR = Path(__file__).parent
MASK_DIR = SITE_DIR / "mask_outputs"
OUTPUT_DIR = SITE_DIR / "NAIP_3DEP_product"
OUTPUT_FILE = OUTPUT_DIR / "pacific_union_college_1m_naip_2022.tif"
TARGET_RES = 1.0  # meters
PAD_M = 50.0  # extra padding around mask extent

# Known outlier files with bad coordinates — exclude from extent calc
OUTLIER_FILES = {
    "CALNU_0031_20250808_1_multiband.tif",
    "CALNU_0032_20250808_1_multiband.tif",
}


def get_multiband_extent():
    """Compute combined bounding box of all multiband TIFs (excluding outliers)."""
    files = sorted(MASK_DIR.glob("*_multiband.tif"))
    xmin, ymin = float("inf"), float("inf")
    xmax, ymax = float("-inf"), float("-inf")
    crs = None

    for f in files:
        if f.name in OUTLIER_FILES:
            continue
        with rasterio.open(f) as src:
            b = src.bounds
            xmin = min(xmin, b.left)
            ymin = min(ymin, b.bottom)
            xmax = max(xmax, b.right)
            ymax = max(ymax, b.top)
            if crs is None:
                crs = src.crs

    xmin -= PAD_M
    ymin -= PAD_M
    xmax += PAD_M
    ymax += PAD_M

    return xmin, ymin, xmax, ymax, crs


def get_naip_urls(bbox_wgs84, year="2022"):
    """Search Planetary Computer STAC for NAIP tiles covering the AOI."""
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    results = catalog.search(
        collections=["naip"],
        bbox=list(bbox_wgs84),
        datetime=f"{year}-01-01/{year}-12-31",
        max_items=10,
    )

    urls = []
    for item in results.items():
        if "image" in item.assets:
            href = item.assets["image"].href
            urls.append((item.id, href))
            print(f"  Found: {item.id}")

    return urls


def download_and_merge_naip(bbox_wgs84, year="2022"):
    """Download NAIP tiles from Planetary Computer, crop to bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    tmp_files = []

    naip_tiles = get_naip_urls(bbox_wgs84, year)
    if not naip_tiles:
        print(f"  No NAIP tiles found for {year}, trying 2020...")
        naip_tiles = get_naip_urls(bbox_wgs84, "2020")

    for tile_id, url in naip_tiles:
        print(f"  Reading: {tile_id}")
        with rasterio.open(url) as src:
            src_bbox = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
            window = from_bounds(*src_bbox, src.transform)
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

            if window.width <= 0 or window.height <= 0:
                print(f"    Tile does not overlap AOI, skipping.")
                continue

            data = src.read(window=window)
            win_transform = src.window_transform(window)

            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp_files.append(tmp.name)
            with rasterio.open(
                tmp.name, "w", driver="GTiff",
                height=data.shape[1], width=data.shape[2], count=src.count,
                dtype=data.dtype, crs=src.crs, transform=win_transform,
            ) as dst:
                dst.write(data)

            print(f"    Read {data.shape[2]}x{data.shape[1]} pixels, {src.count} bands")

    return tmp_files


def main():
    print("Step 1: Computing combined multiband TIF extent...")
    xmin, ymin, xmax, ymax, mask_crs = get_multiband_extent()
    print(f"  Extent (EPSG:6350): ({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})")

    lons, lats = rasterio.warp.transform(
        mask_crs, "EPSG:4326",
        [xmin, xmax, xmin, xmax],
        [ymin, ymin, ymax, ymax],
    )
    bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))
    print(f"  Extent (WGS84): ({bbox_wgs84[0]:.6f}, {bbox_wgs84[1]:.6f}, {bbox_wgs84[2]:.6f}, {bbox_wgs84[3]:.6f})")

    print("\nStep 2: Searching & downloading NAIP tiles from Planetary Computer...")
    tmp_files = download_and_merge_naip(bbox_wgs84)

    if not tmp_files:
        print("ERROR: No tiles downloaded!")
        return

    print(f"\nStep 3: Merging {len(tmp_files)} tiles...")
    src_datasets = [rasterio.open(f) for f in tmp_files]
    merged, merged_transform = merge(src_datasets)
    merged_crs = src_datasets[0].crs
    for ds in src_datasets:
        ds.close()

    print(f"  Merged shape: {merged.shape}")

    print(f"\nStep 4: Reprojecting to {mask_crs} at {TARGET_RES}m resolution...")
    dst_transform, dst_width, dst_height = calculate_default_transform(
        merged_crs, mask_crs,
        merged.shape[2], merged.shape[1],
        left=merged_transform.c,
        bottom=merged_transform.f + merged_transform.e * merged.shape[1],
        right=merged_transform.c + merged_transform.a * merged.shape[2],
        top=merged_transform.f,
        resolution=TARGET_RES,
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

    # Crop to exact mask extent + padding
    crop_window = from_bounds(xmin, ymin, xmax, ymax, dst_transform)
    col_off = max(0, int(crop_window.col_off))
    row_off = max(0, int(crop_window.row_off))
    crop_width = min(int(crop_window.width), dst_width - col_off)
    crop_height = min(int(crop_window.height), dst_height - row_off)

    cropped = dst_data[:, row_off:row_off + crop_height, col_off:col_off + crop_width]
    crop_transform = rasterio.windows.transform(
        rasterio.windows.Window(col_off, row_off, crop_width, crop_height),
        dst_transform,
    )

    print(f"  Output size: {crop_width}x{crop_height}, {cropped.shape[0]} bands, {TARGET_RES}m resolution")

    print(f"\nStep 5: Writing output to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        OUTPUT_FILE, "w", driver="GTiff",
        height=crop_height, width=crop_width,
        count=cropped.shape[0], dtype=cropped.dtype,
        crs=mask_crs, transform=crop_transform,
        compress="deflate",
    ) as dst:
        dst.write(cropped)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.set_band_description(4, "NIR")

    # Cleanup temp files
    for f in tmp_files:
        os.unlink(f)

    print(f"\nDone! Output: {OUTPUT_FILE}")
    print(f"  CRS: {mask_crs}")
    print(f"  Resolution: {TARGET_RES}m")
    print(f"  Bands: 4 (R, G, B, NIR)")
    print(f"  Size: {crop_width} x {crop_height} pixels")


if __name__ == "__main__":
    main()

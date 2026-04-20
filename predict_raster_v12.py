"""
predict_raster_v12.py
=====================
Applies the V12 XGBoost shrub model onto aligned raster files.

V12 differences from V10:
  - Labels from SAM polygon rasterization (not CHM thresholds)
  - No terrain features (elevation/slope/aspect not required)
  - XGBoost instead of LightGBM
  - canopy_in_shrub_range is NOT scaled (binary flag, in no_scale_features)

Required inputs (aligned to the same grid):
  - naip.tif  (4 bands: Red=1, Green=2, Blue=3, NIR=4)
  - chm.tif   (band 1 = canopy height)
"""

import os
import time
import json
import joblib
import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import uniform_filter

MODEL_PATH  = "shrub_classifier_v12.joblib"
SCALER_PATH = "shrub_scaler_v12.joblib"
META_PATH   = "v12_model_meta.json"


def run_shrub_prediction_v12(naip_path, chm_path, output_path,
                              model_path=None, scaler_path=None, meta_path=None):
    # Resolve model paths relative to this file's directory if not absolute
    base_dir = Path(__file__).parent
    model_path  = Path(model_path  or base_dir / MODEL_PATH)
    scaler_path = Path(scaler_path or base_dir / SCALER_PATH)
    meta_path   = Path(meta_path   or base_dir / META_PATH)

    print("Loading V12 XGBoost model...")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)

    features          = meta["features"]
    no_scale_features = set(meta.get("no_scale_features", []))
    threshold         = float(meta.get("cls_threshold", 0.5))
    print(f"  {len(features)} features  no_scale={sorted(no_scale_features)}  threshold={threshold}")

    print("\nReading input rasters...")
    with rasterio.open(naip_path) as src:
        R   = src.read(1).astype(np.float32)
        G   = src.read(2).astype(np.float32)
        B   = src.read(3).astype(np.float32)
        NIR = src.read(4).astype(np.float32)
        profile   = src.profile.copy()
        transform = src.transform

    with rasterio.open(chm_path) as src:
        CHM = src.read(1).astype(np.float32)
        if src.nodata is not None:
            CHM[CHM == src.nodata] = np.nan
    CHM = np.nan_to_num(CHM, nan=0.0).clip(min=0.0)

    for arr in (R, G, B, NIR):
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    rows, cols = R.shape
    print(f"  Raster dimensions: {rows} × {cols} = {rows*cols:,} pixels")

    print("\nComputing spectral indices + canopy features...")
    grids = {
        "naip_red":   R,
        "naip_green": G,
        "naip_blue":  B,
        "naip_nir":   NIR,
        "canopy_height": CHM,
        "ndvi":   (NIR - R)   / (NIR + R   + 1e-8),
        "savi":   1.5 * (NIR - R) / (NIR + R + 0.5 + 1e-8),
        "gndvi":  (NIR - G)   / (NIR + G   + 1e-8),
        "evi2":   2.5 * (NIR - R) / (NIR + 2.4 * R + 1.0 + 1e-8),
        "brightness":      (R + G + B + NIR) / 4.0,
        "greenness_ratio":  G / (R + G + B + 1e-8),
        "canopy_in_shrub_range": ((CHM >= 1.0) & (CHM <= 4.0)).astype(np.float32),
        "canopy_shrub_clipped":  np.clip(CHM, 0.0, 4.0),
    }

    print("Computing multi-scale textures...")
    t0 = time.time()
    texture_sources = ["ndvi", "naip_nir", "naip_red", "naip_green",
                       "savi", "brightness", "canopy_height"]
    needed = set(features)
    for src_name in texture_sources:
        arr = grids[src_name]
        for win in [3, 5, 7]:
            want_mean = f"{src_name}_{win}x{win}mean" in needed
            want_std  = f"{src_name}_{win}x{win}std"  in needed
            if not (want_mean or want_std):
                continue
            mean_val = uniform_filter(arr, size=win, mode="reflect")
            sq_mean  = uniform_filter(arr ** 2, size=win, mode="reflect")
            std_val  = np.sqrt(np.clip(sq_mean - mean_val ** 2, 0, None))
            if want_mean:
                grids[f"{src_name}_{win}x{win}mean"] = mean_val
            if want_std:
                grids[f"{src_name}_{win}x{win}std"]  = std_val
    print(f"  Textures done in {time.time()-t0:.1f}s")

    print("\nBuilding feature matrix...")
    total_px = rows * cols
    X = np.zeros((total_px, len(features)), dtype=np.float32)
    for i, fname in enumerate(features):
        if fname not in grids:
            raise KeyError(f"Feature '{fname}' not in grids. Available: {sorted(grids)}")
        X[:, i] = np.nan_to_num(grids[fname].flatten(), nan=0.0, posinf=0.0, neginf=0.0)

    # Scale only the features that were scaled during training
    scale_idx = [i for i, f in enumerate(features) if f not in no_scale_features]
    X[:, scale_idx] = scaler.transform(X[:, scale_idx])

    print("Running XGBoost inference...")
    t0 = time.time()
    probs = np.zeros(total_px, dtype=np.float32)
    chunk = 500_000
    for start in range(0, total_px, chunk):
        end = min(start + chunk, total_px)
        probs[start:end] = model.predict_proba(X[start:end])[:, 1]
    del X
    print(f"  Inference complete in {time.time()-t0:.1f}s")
    print(f"  Probability stats:  p50={np.percentile(probs,50):.4f}  "
          f"p95={np.percentile(probs,95):.4f}  p99={np.percentile(probs,99):.4f}")

    print(f"\nApplying threshold {threshold}...")
    prob_map = probs.reshape(rows, cols)
    binary   = (prob_map >= threshold).astype(np.uint8)
    n_shrub  = int(binary.sum())
    print(f"  Shrub pixels: {n_shrub:,}  ({100*n_shrub/total_px:.2f}% of raster)")

    # Save probability raster — <prefix>_v12_prob.tif alongside the mask
    _op = str(output_path)
    prob_path = _op.replace("_mask.tif", "_prob.tif") if "_mask.tif" in _op else _op.replace(".tif", "_prob.tif")
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=-1.0, compress="lzw")
    with rasterio.open(prob_path, "w", **prof) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # Save binary mask raster
    prof.update(dtype="uint8", nodata=255)
    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(binary, 1)

    print(f"  Saved binary mask : {output_path}")
    print(f"  Saved probability : {prob_path}")
    print("V12 inference complete.")


def main():
    DIR = "Model_Prediction"
    run_shrub_prediction_v12(
        naip_path=os.path.join(DIR, "naip.tif"),
        chm_path=os.path.join(DIR, "chm.tif"),
        output_path=os.path.join(DIR, "predicted_shrub_v12.tif"),
    )


if __name__ == "__main__":
    main()

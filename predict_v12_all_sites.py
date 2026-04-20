"""
predict_v12_all_sites.py
========================
Run V12 XGBoost pixel classifier on all 6 sites.

For each site:
  1. Load NAIP (4-band) + CHM rasters
  2. Compute the same 55 features used in training
  3. Apply RobustScaler to the 19 scaled features (canopy_in_shrub_range is left as-is)
  4. Predict shrub probability per pixel
  5. Save float32 probability raster + binary mask raster

Usage (WSL):
  python predict_v12_all_sites.py
  python predict_v12_all_sites.py --sites DL_Bliss
  python predict_v12_all_sites.py --threshold 0.544
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
SITES = {
    "Calaveras_Big_trees": {
        "naip": "calaveras_big_trees_1m_naip_2022.tif",
        "chm":  "calaveras_big_trees_canopy_height_1m.tif",
    },
    "DL_Bliss": {
        "naip": "dl_bliss_1m_naip_2022.tif",
        "chm":  "dl_bliss_canopy_height_1m.tif",
    },
    "Independence_Lake": {
        "naip": "independence_lake_1m_naip_2022.tif",
        "chm":  "independence_lake_canopy_height_1m.tif",
    },
    "Pacific_Union": {
        "naip": "pacific_union_college_1m_naip_2022.tif",
        "chm":  "pacific_union_3dep_1m.tif",
    },
    "Sedgwick": {
        "naip": "sedgwick_1m_naip_2022.tif",
        "chm":  "sedgwick_canopy_height_1m.tif",
    },
    "Shaver_Lake": {
        "naip": "shaver_lake_1m_naip_2022.tif",
        "chm":  "shaver_lake_canopy_height_1m.tif",
    },
}

MODEL_PATH  = "./shrub_classifier_v12.joblib"
SCALER_PATH = "./shrub_scaler_v12.joblib"
META_PATH   = "./v12_model_meta.json"


# ---------------------------------------------------------------------------
# Feature computation (matches train_shrub_v12.py exactly)
# ---------------------------------------------------------------------------

def compute_features(naip_path, chm_path):
    with rasterio.open(naip_path) as src:
        R   = src.read(1).astype(np.float32)
        G   = src.read(2).astype(np.float32)
        B   = src.read(3).astype(np.float32)
        NIR = src.read(4).astype(np.float32)
        profile   = src.profile.copy()
        transform = src.transform

    with rasterio.open(chm_path) as src:
        CHM = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None:
            CHM[CHM == nd] = np.nan
    CHM = np.nan_to_num(CHM, nan=0.0).clip(min=0.0)

    # Sanitize NAIP
    for arr in (R, G, B, NIR):
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # Base features
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

    # Multi-scale texture — same bases as training
    texture_sources = ["ndvi", "naip_nir", "naip_red", "naip_green",
                       "savi", "brightness", "canopy_height"]
    for src_name in texture_sources:
        arr = grids[src_name]
        for win in [3, 5, 7]:
            mean_val = uniform_filter(arr, size=win, mode="reflect")
            sq_mean  = uniform_filter(arr ** 2, size=win, mode="reflect")
            std_val  = np.sqrt(np.clip(sq_mean - mean_val ** 2, 0, None))
            grids[f"{src_name}_{win}x{win}mean"] = mean_val
            grids[f"{src_name}_{win}x{win}std"]  = std_val

    return grids, profile, transform


def build_feature_matrix(grids, feature_names):
    H, W = next(iter(grids.values())).shape
    total_px = H * W
    X = np.zeros((total_px, len(feature_names)), dtype=np.float32)
    for i, fname in enumerate(feature_names):
        X[:, i] = np.nan_to_num(grids[fname].flatten(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
    return X, H, W


# ---------------------------------------------------------------------------
# Per-site prediction
# ---------------------------------------------------------------------------

def predict_site(site_name, site_cfg, model, scaler, features, no_scale_features,
                 threshold, out_root):
    print(f"\n{'='*60}")
    print(f"SITE: {site_name}")
    print(f"{'='*60}")

    naip_dir  = Path(site_name) / "NAIP_3DEP_product"
    naip_path = naip_dir / site_cfg["naip"]
    chm_path  = naip_dir / site_cfg["chm"]

    for p in (naip_path, chm_path):
        if not p.exists():
            print(f"  [SKIP] Missing file: {p}")
            return {"site": site_name, "error": f"missing {p.name}"}

    out_dir = Path(out_root) / site_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = site_name.lower().replace(" ", "_")

    # 1. Compute features
    t0 = time.time()
    print(f"  [1/3] Computing features …")
    grids, profile, transform = compute_features(naip_path, chm_path)
    X, H, W = build_feature_matrix(grids, features)
    print(f"        {H}×{W} px = {H*W:,} pixels  ({time.time()-t0:.0f}s)")

    # 2. Scale (exclude no_scale_features)
    t0 = time.time()
    print(f"  [2/3] Scaling + predicting …")
    no_scale_set = set(no_scale_features)
    scale_idx    = [i for i, f in enumerate(features) if f not in no_scale_set]
    X[:, scale_idx] = scaler.transform(X[:, scale_idx])

    # Predict in chunks to avoid OOM
    probs = np.zeros(H * W, dtype=np.float32)
    chunk = 500_000
    for start in range(0, H * W, chunk):
        end = min(start + chunk, H * W)
        probs[start:end] = model.predict_proba(X[start:end])[:, 1]
    del X

    prob_map = probs.reshape(H, W)
    print(f"        done ({time.time()-t0:.0f}s)  "
          f"p50={np.percentile(prob_map, 50):.3f}  "
          f"p95={np.percentile(prob_map, 95):.3f}  "
          f"p99={np.percentile(prob_map, 99):.3f}")

    # 3. Save outputs
    t0 = time.time()
    print(f"  [3/3] Saving rasters …")

    base_prof = profile.copy()
    base_prof.update(count=1, compress="lzw")

    # Probability raster (float32)
    prob_path = out_dir / f"{prefix}_v12_prob.tif"
    base_prof.update(dtype="float32", nodata=-1.0)
    with rasterio.open(prob_path, "w", **base_prof) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # Binary mask (uint8)
    mask_path = out_dir / f"{prefix}_v12_mask.tif"
    base_prof.update(dtype="uint8", nodata=255)
    binary = (prob_map >= threshold).astype(np.uint8)
    with rasterio.open(mask_path, "w", **base_prof) as dst:
        dst.write(binary, 1)

    shrub_px = int(binary.sum())
    shrub_m2 = shrub_px  # 1 m/px → 1 px = 1 m²
    print(f"        Shrub pixels: {shrub_px:,}  ({shrub_m2/1e4:.1f} ha)")
    print(f"        Saved: {prob_path.name}  {mask_path.name}  ({time.time()-t0:.0f}s)")

    return {
        "site":      site_name,
        "size":      f"{H}×{W}",
        "shrub_px":  shrub_px,
        "shrub_ha":  round(shrub_m2 / 1e4, 2),
        "p50":       round(float(np.percentile(prob_map, 50)), 3),
        "p95":       round(float(np.percentile(prob_map, 95)), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sites", nargs="+", default=list(SITES.keys()))
    p.add_argument("--model",     default=MODEL_PATH)
    p.add_argument("--scaler",    default=SCALER_PATH)
    p.add_argument("--meta",      default=META_PATH)
    p.add_argument("--threshold", type=float, default=None,
                   help="Binary threshold (default: from meta cls_threshold)")
    p.add_argument("--out-root",  default="predictions_v12")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading V12 XGBoost model …")
    model  = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    with open(args.meta) as f:
        meta = json.load(f)

    features          = meta["features"]
    no_scale_features = meta.get("no_scale_features", [])
    threshold         = args.threshold if args.threshold is not None else meta["cls_threshold"]

    print(f"  Features: {len(features)}  no_scale: {no_scale_features}  threshold: {threshold}")

    results = []
    t_total = time.time()

    for site_name in args.sites:
        if site_name not in SITES:
            print(f"Unknown site: {site_name}  (valid: {list(SITES)})")
            continue
        try:
            r = predict_site(
                site_name, SITES[site_name],
                model, scaler, features, no_scale_features,
                threshold, args.out_root,
            )
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {site_name}: {exc}")
            traceback.print_exc()
            results.append({"site": site_name, "error": str(exc)})

    print(f"\n{'='*60}")
    print(f"SUMMARY  (total: {(time.time()-t_total)/60:.1f} min)  threshold={threshold}")
    print(f"{'='*60}")
    print(f"{'Site':<25} {'Size':>12} {'Shrub px':>10} {'Shrub ha':>10} {'p50':>6} {'p95':>6}")
    print("-" * 73)
    for r in results:
        if "error" in r:
            print(f"{r['site']:<25}  ERROR: {r['error']}")
        else:
            print(f"{r['site']:<25} {r.get('size',''):>12} {r.get('shrub_px',0):>10,} "
                  f"{r.get('shrub_ha',0):>10.2f} {r.get('p50',0):>6.3f} {r.get('p95',0):>6.3f}")
    print(f"\nOutputs in: {args.out_root}/")


if __name__ == "__main__":
    main()

"""
predict_fuse_all_sites.py
=========================
End-to-end pipeline for all 6 California shrub sites:

  For each site:
    1. Detectron2 sliding-window instance detection (Mask R-CNN, 5-channel)
    2. V10 LightGBM per-pixel probability map
    3. Fusion: rescore + filter + refine masks with v10 agreement
    4. Write per-site outputs to <site>/predictions/

Usage
-----
  # All sites (WSL, venv_linux activated):
  python predict_fuse_all_sites.py

  # Single site:
  python predict_fuse_all_sites.py --sites DL_Bliss

  # Tune fusion thresholds:
  python predict_fuse_all_sites.py --v10-pixel-thr 0.20 --min-coverage 0.03
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import rasterio
import cv2
import torch
import torchvision
import geopandas as gpd
from rasterio.features import rasterize, shapes
from scipy.ndimage import uniform_filter
from shapely.geometry import Point, Polygon, mapping, shape as shp
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
SITES = {
    "Calaveras_Big_trees": {
        "naip": "calaveras_big_trees_1m_naip_2022.tif",
        "dem":  "calaveras_big_trees_3dep_1m.tif",
        "chm":  "calaveras_big_trees_canopy_height_1m.tif",
    },
    "DL_Bliss": {
        "naip": "dl_bliss_1m_naip_2022.tif",
        "dem":  "dl_bliss_3dep_1m.tif",
        "chm":  "dl_bliss_canopy_height_1m.tif",
    },
    "Independence_Lake": {
        "naip": "independence_lake_1m_naip_2022.tif",
        "dem":  "independence_lake_3dep_1m.tif",
        "chm":  "independence_lake_canopy_height_1m.tif",
    },
    "Pacific_Union": {
        "naip": "pacific_union_college_1m_naip_2022.tif",
        "dem":  "pacific_union_3dep_1m.tif",
        "chm":  "pacific_union_canopy_height_1m.tif",
    },
    "Sedgwick": {
        "naip": "sedgwick_1m_naip_2022.tif",
        "dem":  "sedgwick_3dep_1m.tif",
        "chm":  "sedgwick_canopy_height_1m.tif",
    },
    "Shaver_Lake": {
        "naip": "shaver_lake_1m_naip_2022.tif",
        "dem":  "shaver_lake_3dep_1m.tif",
        "chm":  "shaver_lake_canopy_height_1m.tif",
    },
}

# ---------------------------------------------------------------------------
# Fusion parameters
# ---------------------------------------------------------------------------
V10_PIXEL_THR    = 0.30   # pixel is "shrub" in v10 if prob >= this
V10_WEIGHT       = 0.50   # exponent on v10_mean_prob in fused score
MIN_V10_COVERAGE = 0.05   # drop detection if < this fraction agrees with v10
HEIGHT_MIN_M     = 1.0    # CHM height filter: lower bound (m)
HEIGHT_MAX_M     = 40.0   # CHM height filter: upper bound (m)
D2_SCORE_THR     = 0.15   # Detectron2 minimum score threshold
NMS_IOU          = 0.30   # NMS IoU threshold
MIN_REFINED_M2   = 1.0    # drop refined polygons smaller than this (m²)
MAX_BOX_M2       = 200.0  # drop Detectron2 detections whose bbox area exceeds this (m²)
MAX_REFINED_M2   = 150.0  # drop refined polygons larger than this (m²)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
D2_MODEL    = "./model_final_v3/model_final.pth"
V10_MODEL   = "./shrub_classifier_v10.joblib"
V10_SCALER  = "./shrub_scaler_v10.joblib"
V10_META    = "./v10_model_meta.json"
CHAN_STATS  = "./detectron2_dataset/channel_stats.json"


# ===========================================================================
# STEP 1 — Detectron2 inference
# ===========================================================================

def build_detectron2_model(model_path, score_thresh, num_classes=1):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from mask_rcnn_config import ANCHOR_SIZES, ANCHOR_ASPECT_RATIOS

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = ANCHOR_ASPECT_RATIOS

    with open(CHAN_STATS) as f:
        stats = json.load(f)
    cfg.MODEL.PIXEL_MEAN = stats["pixel_mean"]
    cfg.MODEL.PIXEL_STD  = stats["pixel_std"]

    model = build_model(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()
    return model


def run_detectron2(naip_path, chm_path, model, patch_size=128, stride=64,
                   spatial_chunk=1024):
    """
    spatial_chunk: max rows/cols to hold in memory at once.
    Large tiles (e.g. Calaveras 4107x1537) are processed in overlapping
    spatial chunks to avoid OOM. Overlap = patch_size so detections at
    chunk boundaries are not missed.
    """
    from mask_rcnn_config import TRAIN_PATCH_SIZE, UPSCALE

    with rasterio.open(naip_path) as src:
        naip = src.read().astype(np.float32)   # (4, H, W)
        transform = src.transform
        crs = src.crs
    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype(np.float32)
        chm = np.nan_to_num(chm, nan=0.0)

    H, W = naip.shape[1], naip.shape[2]
    img_data = np.concatenate([naip, chm[np.newaxis]], axis=0)  # (5, H, W)
    del naip  # free RAM before GPU inference

    all_boxes, all_scores = [], []
    all_local_masks, all_offsets = [], []
    device = next(model.parameters()).device

    # Build spatial chunk start positions (overlapping by patch_size so no boundary gaps)
    def chunk_starts(total, chunk, step):
        starts = list(range(0, total - patch_size + 1, chunk - patch_size))
        if not starts or starts[-1] + patch_size < total:
            starts.append(max(0, total - chunk))
        return sorted(set(starts))

    cy_starts = chunk_starts(H, spatial_chunk, patch_size)
    cx_starts = chunk_starts(W, spatial_chunk, patch_size)

    for cy0 in cy_starts:
        cy1 = min(H, cy0 + spatial_chunk)
        for cx0 in cx_starts:
            cx1 = min(W, cx0 + spatial_chunk)
            chunk_data = img_data[:, cy0:cy1, cx0:cx1]
            ch, cw = chunk_data.shape[1], chunk_data.shape[2]
            
            chunk_boxes, chunk_scores = [], []
            chunk_local_masks, chunk_offsets = [], []

            for y in range(0, ch - patch_size + 1, stride):
                for x in range(0, cw - patch_size + 1, stride):
                    patch = chunk_data[:, y:y + patch_size, x:x + patch_size]

                    patch_up = torch.nn.functional.interpolate(
                        torch.as_tensor(patch).unsqueeze(0),
                        size=(TRAIN_PATCH_SIZE, TRAIN_PATCH_SIZE),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0).to(device)

                    with torch.no_grad():
                        outputs = model([{"image": patch_up}])[0]

                    inst = outputs["instances"].to("cpu")
                    if len(inst) == 0:
                        continue

                    boxes  = inst.pred_boxes.tensor.numpy() / UPSCALE
                    scores = inst.scores.numpy()
                    masks  = inst.pred_masks.numpy()

                    # Shift from chunk-local to full-image coordinates
                    gx, gy = cx0 + x, cy0 + y
                    boxes[:, [0, 2]] += gx
                    boxes[:, [1, 3]] += gy

                    for i in range(len(boxes)):
                        m_up = torch.as_tensor(masks[i]).float().unsqueeze(0).unsqueeze(0)
                        m_down = torch.nn.functional.interpolate(
                            m_up, size=(patch_size, patch_size),
                            mode="bilinear", align_corners=False,
                        ).squeeze().numpy() > 0.5

                        if m_down.sum() > 0:
                            # Update box to be tight to the local mask
                            ys_m, xs_m = np.where(m_down)
                            boxes[i] = [xs_m.min() + gx, ys_m.min() + gy, 
                                        xs_m.max() + gx, ys_m.max() + gy]

                        chunk_boxes.append(boxes[i])
                        chunk_scores.append(scores[i])
                        chunk_local_masks.append(m_down)
                        chunk_offsets.append((gx, gy))
            
            # Per-chunk NMS to save memory
            if chunk_boxes:
                cb_t = torch.tensor(np.array(chunk_boxes))
                cs_t = torch.tensor(np.array(chunk_scores))
                c_keep = torchvision.ops.nms(cb_t, cs_t, iou_threshold=NMS_IOU).numpy()
                
                for idx in c_keep:
                    all_boxes.append(chunk_boxes[idx])
                    all_scores.append(chunk_scores[idx])
                    all_local_masks.append(chunk_local_masks[idx])
                    all_offsets.append(chunk_offsets[idx])
            
            print(f"        Processed chunk at {cy0},{cx0} - Total detections kept: {len(all_boxes)}")

    if not all_boxes:
        return [], [], [], transform, crs, chm

    # Global NMS on all kept chunk detections
    boxes_t  = torch.tensor(np.array(all_boxes))
    scores_t = torch.tensor(np.array(all_scores))
    keep     = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=NMS_IOU).numpy()

    final_boxes  = np.array(all_boxes)[keep]
    final_scores = np.array(all_scores)[keep]
    
    # Store sparse masks (local patch + offset) to avoid massive full-image arrays
    final_sparse_masks = []
    for idx in keep:
        final_sparse_masks.append({
            "mask":   all_local_masks[idx],
            "offset": all_offsets[idx]
        })

    # Height filter
    valid = []
    for i in range(len(final_sparse_masks)):
        sm = final_sparse_masks[i]
        m_local = sm["mask"]
        gx, gy  = sm["offset"]
        
        # Extract patch from CHM
        patch_chm = chm[gy:gy + patch_size, gx:gx + patch_size]
        
        if m_local.sum() > 0:
            px = patch_chm[m_local > 0]
        else:
            px = patch_chm.ravel()

        if px.size > 0:
            avg_h = float(np.nanmean(px))
            if not (HEIGHT_MIN_M <= avg_h <= HEIGHT_MAX_M):
                continue
            x1, y1, x2, y2 = final_boxes[i]
            box_area = (x2 - x1) * (y2 - y1)  # pixels = m² at 1m/px
            if box_area <= MAX_BOX_M2:
                valid.append(i)

    return (
        final_boxes[valid], final_scores[valid], 
        [final_sparse_masks[i] for i in valid],
        transform, crs, chm,
    )


# ===========================================================================
# STEP 2 — V10 probability map
# ===========================================================================

def calc_texture(grid, win):
    gf = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    mv = (~np.isnan(grid) & ~np.isinf(grid)).astype(np.float32)
    sv  = uniform_filter(gf,      size=win, mode="constant", cval=0.0)
    sm  = uniform_filter(mv,      size=win, mode="constant", cval=0.0)
    sq  = uniform_filter(gf ** 2, size=win, mode="constant", cval=0.0)
    mean = np.where(sm > 0, sv / sm, 0.0)
    var  = np.where(sm > 0, sq / sm - mean ** 2, 0.0)
    return mean, np.sqrt(np.clip(var, 0, None))


def run_v10(naip_path, dem_path, chm_path, model, scaler, features, no_scale_features=None):
    with rasterio.open(naip_path) as src:
        R = src.read(1).astype(np.float32)
        G = src.read(2).astype(np.float32)
        B = src.read(3).astype(np.float32)
        N = src.read(4).astype(np.float32)
        transform = src.transform

    with rasterio.open(dem_path) as src:
        elev  = src.read(1).astype(np.float32)
        slope = src.read(2).astype(np.float32) if src.count >= 2 else None
        asp   = src.read(3).astype(np.float32) if src.count >= 3 else None
        nd = src.nodata
        if nd is not None:
            elev[elev == nd] = np.nan
        if slope is None:
            res_x = transform[0]; res_y = abs(transform[4])
            ef = np.where(np.isnan(elev), np.nanmean(elev), elev).astype(np.float32)
            dy, dx = np.gradient(ef, res_y, res_x)
            slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            asp    = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            chm[chm == nd] = np.nan
    chm = np.nan_to_num(chm, nan=0.0).clip(min=0.0)

    rows, cols = elev.shape
    ndvi  = (N - R) / (N + R + 1e-8)
    savi  = 1.5 * (N - R) / (N + R + 0.5 + 1e-8)
    gndvi = (N - G) / (N + G + 1e-8)
    evi2  = 2.5 * (N - R) / (N + 2.4 * R + 1.0 + 1e-8)
    brt   = (R + G + B + N) / 4.0
    gr    = G / (R + G + B + 1e-8)

    grids = {
        "naip_red": R, "naip_green": G, "naip_blue": B, "naip_nir": N,
        "elevation": elev, "slope": slope, "aspect": asp,
        "canopy_height": chm,
        "ndvi": ndvi, "savi": savi, "gndvi": gndvi, "evi2": evi2,
        "brightness": brt, "greenness_ratio": gr,
        "canopy_in_shrub_range": ((chm >= 1.0) & (chm <= 4.0)).astype(np.float32),
        "canopy_shrub_clipped":  np.clip(chm, 0.0, 4.0),
    }

    needed = set(features)
    for base in ["ndvi","naip_nir","elevation","naip_red","naip_green",
                 "savi","brightness","canopy_height"]:
        for win in [3, 5, 7]:
            want = [(win, s) for s in ("mean","std")
                    if f"{base}_{win}x{win}{s}" in needed]
            if not want:
                continue
            m, s = calc_texture(grids[base], win)
            if (win,"mean") in want: grids[f"{base}_{win}x{win}mean"] = m
            if (win,"std")  in want: grids[f"{base}_{win}x{win}std"]  = s

    total_px = rows * cols
    X = np.zeros((total_px, len(features)), dtype=np.float32)
    for i, fname in enumerate(features):
        X[:, i] = np.nan_to_num(grids[fname].flatten(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
    if no_scale_features:
        no_scale_set = set(no_scale_features)
        scale_idx   = [i for i, f in enumerate(features) if f not in no_scale_set]
        X[:, scale_idx] = scaler.transform(X[:, scale_idx])
        # columns in no_scale_set are left as-is (already 0/1 flags)
    else:
        X = scaler.transform(X)

    probs = np.zeros(total_px, dtype=np.float32)
    chunk = 500_000
    for start in range(0, total_px, chunk):
        end = min(start + chunk, total_px)
        probs[start:end] = model.predict_proba(X[start:end])[:, 1]

    return probs.reshape((rows, cols)), transform


# ===========================================================================
# STEP 3 — Fusion
# ===========================================================================

def poly_to_pixel_mask(geom, transform, shape_hw):
    burned = rasterize(
        [(mapping(geom), 1)],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return burned > 0


def pixels_to_polygon(bool_mask, transform):
    polys = [
        shp(s)
        for s, v in shapes(bool_mask.astype(np.uint8), transform=transform)
        if v == 1
    ]
    return unary_union(polys) if polys else None


def boxes_masks_to_geodataframe(boxes, scores, sparse_masks, raster_transform, crs):
    """Convert sparse Detectron2 outputs to a GeoDataFrame of polygon geometries."""
    rows = []
    for i in range(len(boxes)):
        sm = sparse_masks[i]
        m_local = sm["mask"]
        gx, gy  = sm["offset"]
        
        if m_local.sum() > 0:
            binary = (m_local * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            coords = []
            for c in contours:
                if len(c) >= 3:
                    for pt in c:
                        px_local, py_local = pt[0]
                        # Convert local patch px to global image px
                        px_global, py_global = px_local + gx, py_local + gy
                        # Convert to world coords
                        wx, wy = raster_transform * (float(px_global), float(py_global))
                        coords.append((wx, wy))
                    break
            if coords:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geom = Polygon(coords)
            else:
                geom = _box_to_polygon(boxes[i], raster_transform)
        else:
            geom = _box_to_polygon(boxes[i], raster_transform)

        rows.append({"geometry": geom, "score": float(scores[i])})

    return gpd.GeoDataFrame(rows, crs=crs)


def _box_to_polygon(box, transform):
    x1, y1, x2, y2 = box
    corners = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
    return Polygon([transform * (p[0], p[1]) for p in corners])


def fuse_predictions(gdf, prob_map, raster_transform):
    H, W = prob_map.shape
    v10_mean  = np.zeros(len(gdf))
    v10_cover = np.zeros(len(gdf))
    refined   = []

    for i, (_, row) in enumerate(gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            refined.append(None); continue

        pmask = poly_to_pixel_mask(geom, raster_transform, (H, W))
        n_px  = pmask.sum()
        if n_px == 0:
            refined.append(geom); continue

        inside = prob_map[pmask]
        v10_mean[i]  = float(inside.mean())
        v10_cover[i] = float((inside >= V10_PIXEL_THR).sum() / n_px)

        shrub_px = pmask & (prob_map >= V10_PIXEL_THR)
        if shrub_px.sum() > 0:
            ref = pixels_to_polygon(shrub_px, raster_transform)
            refined.append(ref if ref else geom)
        else:
            refined.append(geom)

    d2_scores   = gdf["score"].values.astype(float)
    fused_scores = d2_scores * (np.clip(v10_mean, 0, 1) ** V10_WEIGHT)
    keep = v10_cover >= MIN_V10_COVERAGE

    out_rows = []
    for i in range(len(gdf)):
        if not keep[i]:
            continue
        rgeom = refined[i]
        if rgeom is None or rgeom.is_empty:
            continue
        if not (MIN_REFINED_M2 <= rgeom.area <= MAX_REFINED_M2):
            continue
        out_rows.append({
            "geometry":      rgeom,
            "d2_score":      float(d2_scores[i]),
            "v10_mean_prob": float(v10_mean[i]),
            "v10_coverage":  float(v10_cover[i]),
            "fused_score":   float(fused_scores[i]),
            "class":         "shrub",
        })

    if not out_rows:
        return gpd.GeoDataFrame(columns=["geometry", "d2_score", "v10_mean_prob", 
                                          "v10_coverage", "fused_score", "class"], 
                                 crs=gdf.crs)

    return gpd.GeoDataFrame(out_rows, crs=gdf.crs)


# ===========================================================================
# Per-site runner
# ===========================================================================

def process_site(site_name, site_cfg, d2_model, v10_model, v10_scaler, features, out_root,
                 no_scale_features=None):
    print(f"\n{'='*60}")
    print(f"SITE: {site_name}")
    print(f"{'='*60}")

    naip_dir = Path(site_name) / "NAIP_3DEP_product"
    naip_path = naip_dir / site_cfg["naip"]
    dem_path  = naip_dir / site_cfg["dem"]
    chm_path  = naip_dir / site_cfg["chm"]

    out_dir = Path(out_root) / site_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = site_name.lower().replace(" ", "_")

    # --- Detectron2 ---
    t0 = time.time()
    print(f"  [1/3] Detectron2 sliding-window inference …")
    boxes, scores, masks, raster_transform, crs, chm = run_detectron2(
        naip_path, chm_path, d2_model
    )
    print(f"        {len(scores)} detections after NMS + height filter  ({time.time()-t0:.0f}s)")

    if len(scores) == 0:
        print("        No detections — skipping site.")
        return {"site": site_name, "d2_raw": 0, "fused": 0}

    gdf_d2 = boxes_masks_to_geodataframe(boxes, scores, masks, raster_transform, crs)

    # Save raw Detectron2 output
    d2_geojson = out_dir / f"{prefix}_detectron2.geojson"
    gdf_d2.to_file(d2_geojson, driver="GeoJSON")

    # --- V10 ---
    t0 = time.time()
    print(f"  [2/3] V10 LightGBM probability map …")
    prob_map, _ = run_v10(naip_path, dem_path, chm_path, v10_model, v10_scaler, features,
                          no_scale_features=no_scale_features)
    print(f"        done  ({time.time()-t0:.0f}s)  "
          f"p50={np.percentile(prob_map,50):.3f}  p99={np.percentile(prob_map,99):.3f}")

    # Save probability raster
    prob_tif = out_dir / f"{prefix}_v10_prob.tif"
    with rasterio.open(naip_path) as src:
        prof = src.profile.copy()
    prof.update(dtype="float32", count=1, nodata=-1, compress="lzw")
    with rasterio.open(prob_tif, "w", **prof) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # --- Fusion ---
    t0 = time.time()
    print(f"  [3/3] Fusing scores (V10_PIXEL_THR={V10_PIXEL_THR}, MIN_COVERAGE={MIN_V10_COVERAGE}) …")
    gdf_fused = fuse_predictions(gdf_d2, prob_map, raster_transform)
    print(f"        {len(gdf_d2)} → {len(gdf_fused)} detections  ({time.time()-t0:.0f}s)")

    if len(gdf_fused) == 0:
        print("        No detections survived fusion.")
        return {"site": site_name, "d2_raw": len(gdf_d2), "fused": 0}

    # Save fused GeoJSON
    fused_geojson = out_dir / f"{prefix}_fused.geojson"
    gdf_fused.to_file(fused_geojson, driver="GeoJSON")

    # Save fused CSV (lat/lon in WGS-84)
    centroids = gdf_fused.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids_wgs = centroids.to_crs("EPSG:4326")
    centroids_wgs["latitude"]  = centroids_wgs.geometry.y
    centroids_wgs["longitude"] = centroids_wgs.geometry.x
    fused_csv = out_dir / f"{prefix}_fused.csv"
    centroids_wgs[["latitude","longitude","d2_score","v10_mean_prob",
                   "v10_coverage","fused_score"]].reset_index(drop=True).to_csv(
        fused_csv, index_label="shrub_id"
    )

    areas = gdf_fused.geometry.area
    print(f"        Polygon area (m2): median={areas.median():.1f}  max={areas.max():.1f}")
    print(f"        Saved: {fused_geojson.name}  {fused_csv.name}  {prob_tif.name}")

    return {
        "site":         site_name,
        "d2_raw":       len(gdf_d2),
        "fused":        len(gdf_fused),
        "area_median":  float(areas.median()),
        "fused_score_mean": float(gdf_fused["fused_score"].mean()),
    }


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sites", nargs="+", default=list(SITES.keys()),
                   help="Site names to process (default: all)")
    p.add_argument("--model",        default=D2_MODEL)
    p.add_argument("--v10-model",    default=V10_MODEL)
    p.add_argument("--v10-scaler",   default=V10_SCALER)
    p.add_argument("--v10-meta",     default=V10_META)
    p.add_argument("--out-root",     default="predictions_all_sites",
                   help="Output root directory")
    p.add_argument("--score-thresh", type=float, default=D2_SCORE_THR)
    p.add_argument("--v10-pixel-thr",type=float, default=V10_PIXEL_THR)
    p.add_argument("--min-coverage", type=float, default=MIN_V10_COVERAGE)
    p.add_argument("--max-box-m2",   type=float, default=MAX_BOX_M2,
                   help="Drop D2 detections with bbox area > this (m²)")
    p.add_argument("--max-refined-m2", type=float, default=MAX_REFINED_M2,
                   help="Drop refined polygons larger than this (m²)")
    p.add_argument("--num-classes", type=int, default=1,
                   help="Number of classes in the Detectron2 model (1=shrub-only, 3=shrub+tree+rock)")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow CLI overrides of module-level constants
    global V10_PIXEL_THR, MIN_V10_COVERAGE, MAX_BOX_M2, MAX_REFINED_M2
    V10_PIXEL_THR    = args.v10_pixel_thr
    MIN_V10_COVERAGE = args.min_coverage
    MAX_BOX_M2       = args.max_box_m2
    MAX_REFINED_M2   = args.max_refined_m2

    # Load shared models once
    print("Loading Detectron2 model …")
    d2_model = build_detectron2_model(args.model, args.score_thresh, args.num_classes)

    print("Loading V10 LightGBM model …")
    v10_model  = joblib.load(args.v10_model)
    v10_scaler = joblib.load(args.v10_scaler)
    with open(args.v10_meta) as f:
        meta = json.load(f)
    features = meta["features"]
    no_scale_features = meta.get("no_scale_features", [])
    print(f"  features: {len(features)}  no_scale: {no_scale_features}  threshold={meta.get('cls_threshold','?')}")

    # Process each site
    results = []
    t_total = time.time()
    for site_name in args.sites:
        if site_name not in SITES:
            print(f"Unknown site: {site_name}  (valid: {list(SITES)})")
            continue
        try:
            r = process_site(
                site_name, SITES[site_name],
                d2_model, v10_model, v10_scaler, features,
                args.out_root,
                no_scale_features=no_scale_features,
            )
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {site_name}: {exc}")
            traceback.print_exc()
            results.append({"site": site_name, "error": str(exc)})

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY  (total time: {(time.time()-t_total)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"{'Site':<25} {'D2 raw':>8} {'Fused':>8} {'Med area m2':>12} {'Mean score':>11}")
    print("-" * 66)
    total_d2 = total_fused = 0
    for r in results:
        if "error" in r:
            print(f"{r['site']:<25}  ERROR: {r['error']}")
            continue
        total_d2    += r.get("d2_raw", 0)
        total_fused += r.get("fused", 0)
        print(f"{r['site']:<25} {r.get('d2_raw',0):>8} {r.get('fused',0):>8} "
              f"{r.get('area_median',0):>12.1f} {r.get('fused_score_mean',0):>11.3f}")
    print("-" * 66)
    print(f"{'TOTAL':<25} {total_d2:>8} {total_fused:>8}")
    print(f"\nOutputs in: {args.out_root}/")


if __name__ == "__main__":
    main()

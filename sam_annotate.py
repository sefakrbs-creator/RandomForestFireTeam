"""
sam_annotate.py
===============
Unsupervised annotation refinement using Segment Anything Model (SAM).

Two passes per patch:

  Pass 1 — Prompted SAM (shrub refinement)
    For each existing circular shrub annotation:
      - Use centroid as a foreground point prompt
      - SAM generates a precise mask fitting the actual shrub shape
      - Replace the circular polygon with the SAM mask polygon
      - Filter: keep only if mean CHM in [CHM_SHRUB_LO, CHM_SHRUB_HI]

  Pass 2 — Automatic Mask Generation (hard negatives)
    Run SAM AMG on every patch (annotated + empty):
      - CHM > CHM_TREE_THR  → label "tree"   (hard negative)
      - CHM < CHM_ROCK_THR  → label "rock"   (hard negative)
      - Overlaps existing shrub mask → skip
      - Area < MIN_AREA_PX  → skip (noise)

Outputs:
  shrub_train_sam.json  — updated train split (refined shrubs + hard negatives)
  shrub_val_sam.json    — updated val split
  sam_stats.json        — per-site annotation statistics

Usage (WSL, venv_linux activated):
  python sam_annotate.py
  python sam_annotate.py --model-type vit_b --checkpoint sam_vit_b_01ec64.pth
"""

import argparse
import json
import copy
import numpy as np
import cv2
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHM_CHANNEL   = 4
RED_CHANNEL   = 0
NIR_CHANNEL   = 3
CHM_SHRUB_LO  = 1.0    # m — minimum shrub height
CHM_SHRUB_HI  = 4.0    # m — maximum shrub height (above = tree)
CHM_TREE_THR  = 4.0    # m — masks above this → "tree" hard negative
CHM_ROCK_THR  = 1.0    # m — masks below this → "rock/ground" hard negative
MIN_AREA_PX   = 4      # px² — discard tiny SAM masks (noise)
MAX_AREA_PX   = 3000   # px² — discard huge masks (entire patch background)
UPSCALE       = 4      # scale patches before SAM (128→512) for better small-obj detection
IOU_OVERLAP   = 0.3    # IoU above this → mask overlaps existing shrub → skip as hard neg

# Category IDs
CAT_SHRUB  = 0
CAT_TREE   = 1
CAT_ROCK   = 2

# AMG settings — tuned for small dense objects
AMG_POINTS_PER_SIDE   = 32    # denser grid for small shrubs
AMG_PRED_IOU_THRESH   = 0.7
AMG_STABILITY_THRESH  = 0.85
AMG_MIN_MASK_REGION   = MIN_AREA_PX * (UPSCALE ** 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_patch(file_name):
    img = np.load(file_name).astype(np.float32)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)


def patch_to_rgb_uint8(img5):
    """Percentile-stretch NAIP R,G,B to uint8 for SAM."""
    rgb = img5[:3].transpose(1, 2, 0).copy()  # (H, W, 3)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        if hi > lo:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo) * 255, 0, 255)
        else:
            rgb[:, :, c] = 0
    return rgb.astype(np.uint8)


def ann_to_mask(ann, H, W):
    """Rasterize a COCO polygon annotation to a binary mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for seg in ann.get("segmentation", []):
        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_to_coco_polygon(binary_mask):
    """Convert a binary mask to COCO polygon segmentation."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    segs = []
    for c in contours:
        if len(c) >= 3:
            flat = c.flatten().tolist()
            if len(flat) >= 6:
                segs.append(flat)
    return segs


def iou(mask_a, mask_b):
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter) / float(union + 1e-8)


def chm_stats(binary_mask, chm):
    """Mean and fraction of shrub-height pixels inside a mask."""
    px = chm[binary_mask]
    if px.size == 0:
        return 0.0, 0.0
    mean_h = float(np.nanmean(px))
    shrub_frac = float(((px >= CHM_SHRUB_LO) & (px <= CHM_SHRUB_HI)).mean())
    return mean_h, shrub_frac


def upscale_rgb(rgb, scale):
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)


def downscale_mask(mask_up, scale, orig_h, orig_w):
    """Downscale an upscaled SAM mask back to native patch coordinates."""
    down = cv2.resize(
        mask_up.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    return down.astype(bool)


def centroid_of_annotation(ann, H, W):
    """Return (x, y) centroid of a COCO annotation in native patch coords."""
    mask = ann_to_mask(ann, H, W)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        # Fall back to bbox center
        x, y, bw, bh = ann["bbox"]
        return x + bw / 2, y + bh / 2
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------------------
# Pass 1 — Prompted SAM: refine existing shrub annotations
# ---------------------------------------------------------------------------

def refine_shrub_annotations(d, predictor, img5):
    """
    Replace circular shrub polygons with SAM-predicted masks.
    Returns updated annotations list.
    """
    H, W = d["height"], d["width"]
    if not d.get("annotations"):
        return []

    rgb = patch_to_rgb_uint8(img5)
    rgb_up = upscale_rgb(rgb, UPSCALE)
    chm = img5[CHM_CHANNEL]

    predictor.set_image(rgb_up)

    refined = []
    for ann in d["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("category_id", CAT_SHRUB) != CAT_SHRUB:
            continue

        cx, cy = centroid_of_annotation(ann, H, W)
        # Scale point to upscaled image
        point = np.array([[cx * UPSCALE, cy * UPSCALE]], dtype=np.float32)
        point_labels = np.array([1])   # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick best mask by score, then validate with CHM
        best = None
        best_score = -1
        for mask_up, score in zip(masks, scores):
            mask_nat = downscale_mask(mask_up, UPSCALE, H, W)
            area = int(mask_nat.sum())
            if area < MIN_AREA_PX or area > MAX_AREA_PX:
                continue
            mean_h, shrub_frac = chm_stats(mask_nat, chm)
            if mean_h < CHM_SHRUB_LO or mean_h > CHM_SHRUB_HI:
                continue
            # Weight score by CHM shrub fraction
            combined = float(score) * (0.5 + 0.5 * shrub_frac)
            if combined > best_score:
                best_score = combined
                best = (mask_nat, area)

        if best is None:
            # SAM couldn't find a valid mask — keep original circular annotation
            refined.append(ann)
            continue

        mask_nat, area = best
        segs = mask_to_coco_polygon(mask_nat)
        if not segs:
            refined.append(ann)
            continue

        ys, xs = np.where(mask_nat)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        refined.append({
            "bbox":        [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            "bbox_mode":   1,
            "segmentation": segs,
            "category_id": CAT_SHRUB,
            "area":        float(area),
            "iscrowd":     0,
            "sam_refined": True,
        })

    return refined


# ---------------------------------------------------------------------------
# Pass 2 — AMG: automatic hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(d, mask_generator, img5, existing_shrub_masks):
    """
    Run SAM AMG and classify non-shrub masks as tree/rock hard negatives.
    Returns list of new hard-negative annotations.
    """
    H, W = d["height"], d["width"]
    chm = img5[CHM_CHANNEL]

    rgb = patch_to_rgb_uint8(img5)
    rgb_up = upscale_rgb(rgb, UPSCALE)

    sam_masks = mask_generator.generate(rgb_up)

    hard_negs = []
    for sm in sam_masks:
        mask_up = sm["segmentation"]
        mask_nat = downscale_mask(mask_up, UPSCALE, H, W)
        area = int(mask_nat.sum())

        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        # Skip if overlaps with any existing shrub annotation
        overlaps = any(
            iou(mask_nat, ex_mask) > IOU_OVERLAP
            for ex_mask in existing_shrub_masks
        )
        if overlaps:
            continue

        mean_h, _ = chm_stats(mask_nat, chm)

        # Classify by height
        if mean_h > CHM_TREE_THR:
            cat = CAT_TREE
        elif mean_h < CHM_ROCK_THR:
            cat = CAT_ROCK
        else:
            continue  # ambiguous height — skip

        segs = mask_to_coco_polygon(mask_nat)
        if not segs:
            continue

        ys, xs = np.where(mask_nat)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        hard_negs.append({
            "bbox":         [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            "bbox_mode":    1,
            "segmentation": segs,
            "category_id":  cat,
            "area":         float(area),
            "iscrowd":      0,
            "hard_negative": True,
            "mean_chm":      float(mean_h),
        })

    return hard_negs


# ---------------------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------------------

def process_split(dicts, predictor, mask_generator, split_name):
    updated = []
    stats = {"refined": 0, "kept_original": 0, "trees": 0, "rocks": 0, "skipped": 0}

    for i, d in enumerate(dicts):
        d = copy.deepcopy(d)
        H, W = d["height"], d["width"]

        try:
            img5 = load_patch(d["file_name"])
        except Exception as e:
            print(f"  [WARN] could not load {d['file_name']}: {e}")
            updated.append(d)
            stats["skipped"] += 1
            continue

        # Pass 1: refine shrub annotations
        if d.get("annotations"):
            refined = refine_shrub_annotations(d, predictor, img5)
            n_refined = sum(1 for a in refined if a.get("sam_refined"))
            n_kept    = len(refined) - n_refined
            stats["refined"]       += n_refined
            stats["kept_original"] += n_kept
        else:
            refined = []

        # Build existing shrub mask set for overlap check
        existing_masks = [ann_to_mask(a, H, W) for a in refined
                          if a.get("category_id", CAT_SHRUB) == CAT_SHRUB]

        # Pass 2: hard negative mining
        hard_negs = mine_hard_negatives(d, mask_generator, img5, existing_masks)
        stats["trees"] += sum(1 for a in hard_negs if a["category_id"] == CAT_TREE)
        stats["rocks"] += sum(1 for a in hard_negs if a["category_id"] == CAT_ROCK)

        d["annotations"] = refined + hard_negs
        updated.append(d)

        if (i + 1) % 50 == 0 or (i + 1) == len(dicts):
            print(f"  [{split_name}] {i+1}/{len(dicts)}  "
                  f"refined={stats['refined']}  trees={stats['trees']}  rocks={stats['rocks']}")

    return updated, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="./sam_vit_h_4b8939.pth")
    p.add_argument("--model-type",   default="vit_h",
                   choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--train-json",   default="./detectron2_dataset/shrub_train.json")
    p.add_argument("--val-json",     default="./detectron2_dataset/shrub_val.json")
    p.add_argument("--out-train",    default="./detectron2_dataset/shrub_train_sam.json")
    p.add_argument("--out-val",      default="./detectron2_dataset/shrub_val_sam.json")
    p.add_argument("--sites",        nargs="+", default=None,
                   help="Process only these sites (default: all)")
    p.add_argument("--skip-amg",     action="store_true",
                   help="Skip hard-negative AMG pass (faster, only refine shrubs)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading SAM {args.model_type} from {args.checkpoint} …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    print(f"  SAM loaded on {device}")

    predictor = SamPredictor(sam)

    if not args.skip_amg:
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=AMG_POINTS_PER_SIDE,
            pred_iou_thresh=AMG_PRED_IOU_THRESH,
            stability_score_thresh=AMG_STABILITY_THRESH,
            min_mask_region_area=AMG_MIN_MASK_REGION,
        )
    else:
        mask_generator = None

    with open(args.train_json) as f:
        train = json.load(f)
    with open(args.val_json) as f:
        val = json.load(f)

    if args.sites:
        train = [d for d in train if d["site"] in args.sites]
        val   = [d for d in val   if d["site"] in args.sites]
        print(f"Filtering to sites: {args.sites}")

    # Filter predicted_mask.tif noise
    before_train = len(train)
    train = [d for d in train if d.get("source_mask") != "predicted_mask.tif"
             or not d.get("annotations")]
    print(f"Filtered predicted_mask.tif noise: {before_train} → {len(train)} train tiles")

    all_stats = {}

    print(f"\n--- Processing TRAIN ({len(train)} tiles) ---")
    train_updated, train_stats = process_split(
        train, predictor, mask_generator, "train"
    )
    all_stats["train"] = train_stats

    print(f"\n--- Processing VAL ({len(val)} tiles) ---")
    val_updated, val_stats = process_split(
        val, predictor, mask_generator, "val"
    )
    all_stats["val"] = val_stats

    # Save
    with open(args.out_train, "w") as f:
        json.dump(train_updated, f)
    with open(args.out_val, "w") as f:
        json.dump(val_updated, f)

    stats_path = Path("./detectron2_dataset/sam_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAIN: refined={train_stats['refined']}  kept_orig={train_stats['kept_original']}  "
          f"trees={train_stats['trees']}  rocks={train_stats['rocks']}")
    print(f"VAL:   refined={val_stats['refined']}  kept_orig={val_stats['kept_original']}  "
          f"trees={val_stats['trees']}  rocks={val_stats['rocks']}")
    print(f"Saved: {args.out_train}")
    print(f"Saved: {args.out_val}")
    print(f"\nNext: retrain v5 with --train-json detectron2_dataset/shrub_train_sam.json")


if __name__ == "__main__":
    main()

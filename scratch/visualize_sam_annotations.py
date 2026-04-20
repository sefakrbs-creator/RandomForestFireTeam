"""
visualize_sam_annotations.py
============================
Visualize SAM-refined shrub annotations from shrub_train_sam.json / shrub_val_sam.json.

Shows:
  - Original NAIP RGB patch (R,G,B from channels 0,1,2)
  - Shrub masks     (green)
  - Tree hard-negs  (red)
  - Rock hard-negs  (blue)
  - SAM-refined vs kept-original shrubs marked differently

Usage:
  python scratch/visualize_sam_annotations.py                  # random 12 patches from train
  python scratch/visualize_sam_annotations.py --split val      # from val
  python scratch/visualize_sam_annotations.py --site independence_lake
  python scratch/visualize_sam_annotations.py --n 20 --only-refined
  python scratch/visualize_sam_annotations.py --save-dir scratch/sam_viz
"""

import argparse
import json
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to TkAgg/Qt5Agg if you have a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ─── colour palette ──────────────────────────────────────────────────────────
COLOURS = {
    0: (0.1, 0.9, 0.2),   # shrub  → green
    1: (0.9, 0.1, 0.1),   # tree   → red
    2: (0.2, 0.4, 1.0),   # rock   → blue
}
ALPHA_MASK  = 0.45
ALPHA_EDGE  = 0.9


# ─── helpers ──────────────────────────────────────────────────────────────────

def percentile_stretch(arr, lo=2, hi=98):
    """Stretch a float array to [0,1] using percentile clipping."""
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    if p_hi > p_lo:
        return np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1)
    return np.zeros_like(arr)


def load_rgb(file_name):
    """Load .npy patch and return uint8 RGB (H,W,3)."""
    img = np.load(file_name).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    # NAIP channels: R=0, G=1, B=2  (NIR=3, CHM=4)
    r = percentile_stretch(img[0])
    g = percentile_stretch(img[1])
    b = percentile_stretch(img[2])
    return np.stack([r, g, b], axis=-1)   # (H,W,3) float [0,1]


def draw_annotations(ax, rgb, annotations, H, W):
    """Overlay all annotations onto ax."""
    ax.imshow(rgb, interpolation="nearest")

    overlay = np.zeros((H, W, 4), dtype=np.float32)

    for ann in annotations:
        cat = ann.get("category_id", 0)
        col = COLOURS.get(cat, (1, 1, 0))
        is_refined = ann.get("sam_refined", False)

        # Rasterize polygon
        segs = ann.get("segmentation", [])
        if not segs:
            continue
        mask = np.zeros((H, W), dtype=np.uint8)
        for seg in segs:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)

        # Fill colour
        overlay[mask == 1, :3] = col
        overlay[mask == 1,  3] = ALPHA_MASK

        # Edge highlight — thicker for SAM-refined shrubs
        edge = cv2.Canny(mask * 255, 50, 150)
        if is_refined:
            edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
        overlay[edge > 0, :3] = col
        overlay[edge > 0,  3] = ALPHA_EDGE

    ax.imshow(overlay, interpolation="nearest")


def make_legend():
    return [
        mpatches.Patch(color=COLOURS[0], label="Shrub (kept orig)", alpha=0.6),
        mpatches.Patch(color=COLOURS[0], label="Shrub (SAM-refined)",
                       alpha=0.9, linewidth=2, linestyle="--"),
        mpatches.Patch(color=COLOURS[1], label="Tree (hard-neg)", alpha=0.6),
        mpatches.Patch(color=COLOURS[2], label="Rock (hard-neg)", alpha=0.6),
    ]


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",       default="train", choices=["train", "val"])
    p.add_argument("--json-dir",    default="./detectron2_dataset")
    p.add_argument("--n",           type=int, default=12, help="Number of patches to show")
    p.add_argument("--site",        default=None, help="Filter to one site")
    p.add_argument("--only-refined",action="store_true",
                   help="Only show patches that have ≥1 SAM-refined shrub")
    p.add_argument("--only-negatives", action="store_true",
                   help="Only show patches that have hard negatives")
    p.add_argument("--save-dir",    default="scratch/sam_viz",
                   help="Directory to save PNG files (default: scratch/sam_viz)")
    p.add_argument("--cols",        type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    json_path = Path(args.json_dir) / f"shrub_{args.split}_sam.json"
    print(f"Loading {json_path} …")
    with open(json_path) as f:
        dicts = json.load(f)

    print(f"  Total patches in split: {len(dicts)}")

    # ── Filters ──────────────────────────────────────────────────────────────
    if args.site:
        dicts = [d for d in dicts if d.get("site") == args.site]
        print(f"  After site filter '{args.site}': {len(dicts)}")

    if args.only_refined:
        dicts = [d for d in dicts
                 if any(a.get("sam_refined") for a in d.get("annotations", []))]
        print(f"  After only-refined filter: {len(dicts)}")

    if args.only_negatives:
        dicts = [d for d in dicts
                 if any(a.get("hard_negative") for a in d.get("annotations", []))]
        print(f"  After only-negatives filter: {len(dicts)}")

    if not dicts:
        print("No patches match your filters — exiting.")
        return

    # Randomly sample
    sample = random.sample(dicts, min(args.n, len(dicts)))

    # ── Print quick stats per sampled patch ──────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"{'Patch':<40} {'Shrubs':>7} {'SAMref':>7} {'Trees':>7} {'Rocks':>7}")
    print(f"{'-'*60}")
    for d in sample:
        anns = d.get("annotations", [])
        shrubs  = sum(1 for a in anns if a.get("category_id", 0) == 0)
        refined = sum(1 for a in anns if a.get("sam_refined"))
        trees   = sum(1 for a in anns if a.get("category_id") == 1)
        rocks   = sum(1 for a in anns if a.get("category_id") == 2)
        name = Path(d["file_name"]).name
        print(f"  {name:<38} {shrubs:>7} {refined:>7} {trees:>7} {rocks:>7}")
    print(f"{'-'*60}\n")

    # ── Build figure ──────────────────────────────────────────────────────────
    cols = min(args.cols, len(sample))
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, d in enumerate(sample):
        ax = axes[i]
        file_name = d["file_name"]
        H, W = d.get("height", 128), d.get("width", 128)
        anns = d.get("annotations", [])

        try:
            rgb = load_rgb(file_name)
        except Exception as e:
            ax.set_title(f"LOAD ERR\n{Path(file_name).name}", fontsize=7)
            ax.axis("off")
            continue

        draw_annotations(ax, rgb, anns, H, W)

        # Title: summary stats
        shrubs  = sum(1 for a in anns if a.get("category_id", 0) == 0)
        refined = sum(1 for a in anns if a.get("sam_refined"))
        trees   = sum(1 for a in anns if a.get("category_id") == 1)
        rocks   = sum(1 for a in anns if a.get("category_id") == 2)
        site    = d.get("site", "?")
        name    = Path(file_name).stem[-20:]

        title = (f"{name}\n"
                 f"site={site}  shrubs={shrubs}(↑{refined})  "
                 f"trees={trees}  rocks={rocks}")
        ax.set_title(title, fontsize=6.5, pad=3)
        ax.axis("off")

    # Hide unused axes
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    fig.legend(handles=make_legend(), loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"SAM Annotations — {args.split} split  ({len(sample)} patches)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    # Save grid
    grid_path = save_dir / f"sam_{args.split}_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid → {grid_path}")
    plt.close(fig)

    # ── Also save each patch individually ─────────────────────────────────────
    print("Saving individual patches …")
    for d in sample:
        file_name = d["file_name"]
        H, W = d.get("height", 128), d.get("width", 128)
        anns = d.get("annotations", [])
        try:
            rgb = load_rgb(file_name)
        except Exception:
            continue

        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
        draw_annotations(ax2, rgb, anns, H, W)
        anns_str = (f"shrubs={sum(1 for a in anns if a.get('category_id',0)==0)} "
                    f"trees={sum(1 for a in anns if a.get('category_id')==1)} "
                    f"rocks={sum(1 for a in anns if a.get('category_id')==2)}")
        ax2.set_title(f"{Path(file_name).stem}\n{anns_str}", fontsize=7)
        ax2.axis("off")
        out = save_dir / f"{Path(file_name).stem}.png"
        fig2.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig2)

    print(f"\nDone. All images saved to: {save_dir.resolve()}")
    print("Open the grid with:  start scratch\\sam_viz\\sam_train_grid.png")


if __name__ == "__main__":
    main()

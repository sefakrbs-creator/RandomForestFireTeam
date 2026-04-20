"""
generate_all_sam_viz.py
=======================
Generate PNG visualizations for ALL SAM-annotated patches (train + val),
organized by site.

Outputs:
  sam_viz_all/<site>/           — individual tile PNGs per site
  sam_viz_all/<site>_grid.png   — per-site overview grid
  sam_viz_all/summary_grid.png  — cross-site summary (random 24 patches)

Usage:
  python generate_all_sam_viz.py
  python generate_all_sam_viz.py --out-dir my_viz --splits train
  python generate_all_sam_viz.py --dpi 100 --no-individual
"""

import argparse
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── colour palette ─────────────────────────────────────────────────────────────
COLOURS = {
    0: (0.1, 0.9, 0.2),   # shrub → green
    1: (0.9, 0.1, 0.1),   # tree  → red
    2: (0.2, 0.4, 1.0),   # rock  → blue
}
ALPHA_MASK = 0.45
ALPHA_EDGE = 0.90


def percentile_stretch(arr, lo=2, hi=98):
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    if p_hi > p_lo:
        return np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1)
    return np.zeros_like(arr)


def load_rgb(file_name):
    img = np.load(file_name).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.stack([percentile_stretch(img[c]) for c in range(3)], axis=-1)


def draw_annotations(ax, rgb, annotations, H, W):
    ax.imshow(rgb, interpolation="nearest")
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for ann in annotations:
        cat = ann.get("category_id", 0)
        col = COLOURS.get(cat, (1, 1, 0))
        for seg in ann.get("segmentation", []):
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            overlay[mask == 1, :3] = col
            overlay[mask == 1,  3] = ALPHA_MASK
            edge = cv2.Canny(mask * 255, 50, 150)
            if ann.get("sam_refined"):
                edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
            overlay[edge > 0, :3] = col
            overlay[edge > 0,  3] = ALPHA_EDGE
    ax.imshow(overlay, interpolation="nearest")


def ann_counts(anns):
    shrubs  = sum(1 for a in anns if a.get("category_id", 0) == 0)
    refined = sum(1 for a in anns if a.get("sam_refined"))
    trees   = sum(1 for a in anns if a.get("category_id") == 1)
    rocks   = sum(1 for a in anns if a.get("category_id") == 2)
    return shrubs, refined, trees, rocks


def legend_handles():
    return [
        mpatches.Patch(color=COLOURS[0], label="Shrub (kept orig)",    alpha=0.6),
        mpatches.Patch(color=COLOURS[0], label="Shrub (SAM-refined)",  alpha=0.9),
        mpatches.Patch(color=COLOURS[1], label="Tree (hard-neg)",      alpha=0.6),
        mpatches.Patch(color=COLOURS[2], label="Rock (hard-neg)",      alpha=0.6),
    ]


def save_individual(d, out_dir, dpi):
    file_name = d["file_name"]
    H, W = d.get("height", 128), d.get("width", 128)
    anns = d.get("annotations", [])
    try:
        rgb = load_rgb(file_name)
    except Exception as e:
        print(f"    [SKIP] {Path(file_name).name}: {e}")
        return None
    shrubs, refined, trees, rocks = ann_counts(anns)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    draw_annotations(ax, rgb, anns, H, W)
    ax.set_title(
        f"{Path(file_name).stem}\nshrubs={shrubs}(↑{refined})  trees={trees}  rocks={rocks}",
        fontsize=6.5, pad=3
    )
    ax.axis("off")
    out_path = out_dir / f"{Path(file_name).stem}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_site_grid(site_dicts, site, out_path, dpi, ncols=6):
    n = len(site_dicts)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.7))
    axes = np.array(axes).flatten()
    for i, d in enumerate(site_dicts):
        ax = axes[i]
        H, W = d.get("height", 128), d.get("width", 128)
        anns = d.get("annotations", [])
        try:
            rgb = load_rgb(d["file_name"])
        except Exception:
            ax.axis("off")
            continue
        draw_annotations(ax, rgb, anns, H, W)
        shrubs, refined, trees, rocks = ann_counts(anns)
        ax.set_title(
            f"{Path(d['file_name']).stem[-18:]}\ns={shrubs}(↑{refined}) t={trees} r={rocks}",
            fontsize=5.5, pad=2
        )
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4,
               fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{site}  —  {n} patches", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {out_path.name}  ({n} patches)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir",      default="./detectron2_dataset")
    p.add_argument("--out-dir",       default="sam_viz_all")
    p.add_argument("--splits",        nargs="+", default=["train", "val"])
    p.add_argument("--dpi",           type=int, default=100)
    p.add_argument("--no-individual", action="store_true",
                   help="Skip per-tile PNGs, only save grids")
    p.add_argument("--grid-cols",     type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_dicts = []
    for split in args.splits:
        json_path = Path(args.json_dir) / f"shrub_{split}_sam.json"
        print(f"Loading {json_path} …")
        with open(json_path) as f:
            dicts = json.load(f)
        for d in dicts:
            d["_split"] = split
        all_dicts.extend(dicts)
        print(f"  {split}: {len(dicts)} patches")

    print(f"\nTotal: {len(all_dicts)} patches\n")

    # Group by site
    sites = sorted(set(d.get("site", "unknown") for d in all_dicts))
    print(f"Sites: {sites}\n")

    t_total = time.time()
    total_saved = 0

    for site in sites:
        site_dicts = [d for d in all_dicts if d.get("site") == site]
        site_dir   = out_root / site
        site_dir.mkdir(exist_ok=True)

        print(f"[{site}]  {len(site_dicts)} patches")

        # Individual PNGs
        if not args.no_individual:
            t0 = time.time()
            for j, d in enumerate(site_dicts):
                save_individual(d, site_dir, args.dpi)
                if (j + 1) % 50 == 0:
                    print(f"  {j+1}/{len(site_dicts)} tiles saved …")
            total_saved += len(site_dicts)
            print(f"  Individual PNGs done in {time.time()-t0:.0f}s")

        # Site grid
        grid_path = out_root / f"{site}_grid.png"
        save_site_grid(site_dicts, site, grid_path, args.dpi, ncols=args.grid_cols)

    # Summary grid (up to 24 patches, 4 per site)
    import random
    random.seed(42)
    summary = []
    for site in sites:
        site_dicts = [d for d in all_dicts if d.get("site") == site]
        summary.extend(random.sample(site_dicts, min(4, len(site_dicts))))

    save_site_grid(summary, "All Sites (4 per site)", out_root / "summary_grid.png",
                   args.dpi, ncols=args.grid_cols)

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Individual tiles : {total_saved}")
    print(f"Output directory : {out_root.resolve()}")


if __name__ == "__main__":
    main()

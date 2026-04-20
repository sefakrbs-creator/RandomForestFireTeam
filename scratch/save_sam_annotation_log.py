"""
save_sam_annotation_log.py
==========================
Reads shrub_train_sam.json and shrub_val_sam.json and writes a detailed
human-readable log file: sam_annotation_log.txt

Log contains:
  - Global summary (total counts, refinement rates, class breakdown)
  - Per-site summary table
  - Per-patch detail rows (one line each)
  - CHM validity proxy stats for SAM-refined shrubs

Usage:
  python scratch/save_sam_annotation_log.py
  python scratch/save_sam_annotation_log.py --out my_log.txt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


CAT_NAMES = {0: "shrub", 1: "tree", 2: "rock"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ann_stats(annotations):
    """Return per-category counts + SAM refinement info."""
    counts = defaultdict(int)
    refined = 0
    kept_orig = 0
    hard_neg = 0
    areas = defaultdict(list)
    chm_vals = []

    for a in annotations:
        cat = a.get("category_id", 0)
        counts[cat] += 1
        areas[cat].append(a.get("area", 0))
        if a.get("sam_refined"):
            refined += 1
        if a.get("hard_negative"):
            hard_neg += 1
        else:
            kept_orig += 1 if not a.get("sam_refined") else 0
        if "mean_chm" in a:
            chm_vals.append(a["mean_chm"])

    return counts, refined, kept_orig, hard_neg, areas, chm_vals


def process_split(dicts, split_name):
    """Aggregate stats for one split and return (rows, global_stats, per_site_stats)."""
    rows = []
    global_stats = defaultdict(int)
    per_site = defaultdict(lambda: defaultdict(int))

    for d in dicts:
        anns = d.get("annotations", [])
        site = d.get("site", "unknown")
        patch = Path(d["file_name"]).name

        counts, refined, kept_orig, hard_neg, areas, chm_vals = ann_stats(anns)

        n_shrub = counts[0]
        n_tree  = counts[1]
        n_rock  = counts[2]
        n_total = sum(counts.values())

        # Per-patch row
        mean_shrub_area = float(np.mean(areas[0])) if areas[0] else 0.0
        mean_chm = float(np.mean(chm_vals)) if chm_vals else float("nan")

        rows.append({
            "split":          split_name,
            "site":           site,
            "patch":          patch,
            "total_ann":      n_total,
            "shrubs":         n_shrub,
            "sam_refined":    refined,
            "kept_orig":      kept_orig,
            "trees":          n_tree,
            "rocks":          n_rock,
            "hard_neg":       hard_neg,
            "mean_shrub_area_px": mean_shrub_area,
            "mean_tree_chm":  mean_chm,
        })

        # Aggregate
        global_stats["patches"]      += 1
        global_stats["total_ann"]    += n_total
        global_stats["shrubs"]       += n_shrub
        global_stats["sam_refined"]  += refined
        global_stats["kept_orig"]    += kept_orig
        global_stats["trees"]        += n_tree
        global_stats["rocks"]        += n_rock
        global_stats["hard_neg"]     += hard_neg

        # Per site
        per_site[site]["patches"]     += 1
        per_site[site]["shrubs"]      += n_shrub
        per_site[site]["sam_refined"] += refined
        per_site[site]["trees"]       += n_tree
        per_site[site]["rocks"]       += n_rock

    return rows, global_stats, per_site


def write_log(train_rows, train_stats, train_site,
              val_rows,   val_stats,   val_site,
              out_path):

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    def ln(s=""):
        lines.append(s)

    # ── Header ────────────────────────────────────────────────────────────────
    ln("=" * 80)
    ln("  SAM ANNOTATION LOG")
    ln(f"  Generated : {ts}")
    ln(f"  Train JSON: detectron2_dataset/shrub_train_sam.json")
    ln(f"  Val JSON  : detectron2_dataset/shrub_val_sam.json")
    ln("=" * 80)
    ln()

    # ── Global Summary ─────────────────────────────────────────────────────────
    for split_name, stats in [("TRAIN", train_stats), ("VAL", val_stats)]:
        n_shrubs   = stats["shrubs"]
        n_refined  = stats["sam_refined"]
        n_kept     = stats["kept_orig"]
        refinement_rate = (n_refined / n_shrubs * 100) if n_shrubs else 0.0
        total_ann  = stats["total_ann"]
        shrub_pct  = (n_shrubs / total_ann * 100) if total_ann else 0.0

        ln(f"[{split_name}] GLOBAL SUMMARY")
        ln("-" * 60)
        ln(f"  Patches              : {stats['patches']:>8,}")
        ln(f"  Total annotations    : {total_ann:>8,}")
        ln(f"  -- Shrubs (total)    : {n_shrubs:>8,}  ({shrub_pct:.1f}% of all ann)")
        ln(f"       SAM-refined     : {n_refined:>8,}  ({refinement_rate:.1f}% of shrubs)")
        ln(f"       Kept original   : {n_kept:>8,}")
        ln(f"  -- Trees (hard-neg)  : {stats['trees']:>8,}")
        ln(f"  -- Rocks (hard-neg)  : {stats['rocks']:>8,}")
        ln(f"  -- Total hard-neg    : {stats['hard_neg']:>8,}")
        ln()

    # ── Per-Site Summary Table ─────────────────────────────────────────────────
    for split_name, site_dict in [("TRAIN", train_site), ("VAL", val_site)]:
        ln(f"[{split_name}] PER-SITE BREAKDOWN")
        ln("-" * 80)
        hdr = f"  {'Site':<35} {'Patches':>8} {'Shrubs':>8} {'SAMref':>8} {'Trees':>8} {'Rocks':>8}"
        ln(hdr)
        ln("  " + "-" * 76)
        for site in sorted(site_dict):
            s = site_dict[site]
            ref_pct = (s["sam_refined"] / s["shrubs"] * 100) if s["shrubs"] else 0.0
            ln(f"  {site:<35} {s['patches']:>8} {s['shrubs']:>8} "
               f"{s['sam_refined']:>7}({ref_pct:4.0f}%) {s['trees']:>8} {s['rocks']:>8}")
        ln()

    # ── Per-Patch Detail ──────────────────────────────────────────────────────
    for split_name, rows in [("TRAIN", train_rows), ("VAL", val_rows)]:
        ln(f"[{split_name}] PER-PATCH DETAIL  ({len(rows)} patches)")
        ln("-" * 100)
        hdr = (f"  {'Patch':<45} {'Site':<25} "
               f"{'Shrubs':>7} {'SAMref':>7} {'Trees':>7} {'Rocks':>7} {'AvgArea':>8}")
        ln(hdr)
        ln("  " + "-" * 96)
        for r in sorted(rows, key=lambda x: (x["site"], x["patch"])):
            area_str = f"{r['mean_shrub_area_px']:7.1f}" if r["shrubs"] else "    n/a"
            ln(f"  {r['patch']:<45} {r['site']:<25} "
               f"{r['shrubs']:>7} {r['sam_refined']:>7} "
               f"{r['trees']:>7} {r['rocks']:>7} {area_str:>8}")
        ln()

    # ── Annotation Quality Proxies ─────────────────────────────────────────────
    ln("ANNOTATION QUALITY PROXIES")
    ln("-" * 60)

    for split_name, rows in [("TRAIN", train_rows), ("VAL", val_rows)]:
        # Patches with >=1 shrub
        shrub_patches = [r for r in rows if r["shrubs"] > 0]
        # Patches with >=1 SAM-refined shrub
        refined_patches = [r for r in rows if r["sam_refined"] > 0]
        # Patches with zero shrubs (empty negatives)
        empty_patches = [r for r in rows if r["shrubs"] == 0]

        ln(f"  [{split_name}]")
        ln(f"    Patches with >=1 shrub       : {len(shrub_patches):>6}")
        ln(f"    Patches with SAM-refined shrub: {len(refined_patches):>6}")
        ln(f"    Empty patches (no shrubs)    : {len(empty_patches):>6}")

        all_shrub_areas = [r["mean_shrub_area_px"] for r in rows if r["shrubs"] > 0]
        if all_shrub_areas:
            ln(f"    Mean shrub mask area (px2)   : {np.mean(all_shrub_areas):>8.1f}")
            ln(f"    Median shrub mask area (px2) : {np.median(all_shrub_areas):>8.1f}")
            ln(f"    Min / Max shrub area (px2)   : {np.min(all_shrub_areas):>8.1f} / {np.max(all_shrub_areas):.1f}")

        # Class balance ratio
        t_stats = train_stats if split_name == "TRAIN" else val_stats
        shrub_n = t_stats["shrubs"]
        tree_n  = t_stats["trees"]
        rock_n  = t_stats["rocks"]
        if shrub_n > 0:
            ln(f"    Class ratio tree:shrub       : {tree_n/shrub_n:>8.1f}:1")
            ln(f"    Class ratio rock:shrub       : {rock_n/shrub_n:>8.1f}:1")
            ln(f"    Class ratio (tree+rock):shrub: {(tree_n+rock_n)/shrub_n:>8.1f}:1")
        ln()

    ln("=" * 80)
    ln("END OF LOG")
    ln("=" * 80)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default="./detectron2_dataset")
    p.add_argument("--out",      default="./sam_annotation_log.txt")
    return p.parse_args()


def main():
    args = parse_args()
    json_dir = Path(args.json_dir)

    print("Loading train JSON...")
    train = load_json(json_dir / "shrub_train_sam.json")
    print("Loading val JSON...")
    val   = load_json(json_dir / "shrub_val_sam.json")

    print("Processing train split...")
    train_rows, train_stats, train_site = process_split(train, "train")
    print("Processing val split...")
    val_rows,   val_stats,   val_site   = process_split(val,   "val")

    out_path = Path(args.out)
    print(f"Writing log to {out_path} ...")
    text = write_log(train_rows, train_stats, train_site,
                     val_rows,   val_stats,   val_site,
                     out_path)

    # Print summary to console too
    print("\n" + "=" * 60)
    for line in text.split("\n")[:50]:   # first 50 lines preview
        print(line)
    print("...")
    print(f"\nFull log saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

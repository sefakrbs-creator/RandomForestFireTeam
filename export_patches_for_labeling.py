"""
export_patches_for_labeling.py
==============================
Exports all dataset patches as PNG images (RGB + CHM overlay) organized by site,
ready for import into Label Studio for false-positive hard-negative annotation.

Outputs:
  label_studio_images/<site>/  — RGB PNGs (NAIP R,G,B)
  label_studio_images/<site>_chm/  — CHM heatmap PNGs (height context)
  label_studio_import.json  — Label Studio pre-annotation import file

Usage:
  python export_patches_for_labeling.py
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import colorsys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_JSON = "detectron2_dataset/shrub_train.json"
VAL_JSON   = "detectron2_dataset/shrub_val.json"
OUT_DIR    = Path("label_studio_images")
CHM_CHANNEL = 4
CHM_MAX_M   = 10.0   # clip CHM at this value for visualization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stretch(arr, lo_pct=2, hi_pct=98):
    """Percentile stretch to [0, 255]."""
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def chm_to_colormap(chm, max_m=CHM_MAX_M):
    """CHM → false-color heatmap: blue(0m) → green(2m) → red(>4m)."""
    H, W = chm.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    norm = np.clip(chm / max_m, 0, 1)
    for i in range(H):
        for j in range(W):
            # hue: 0.67 (blue) → 0.0 (red) as height increases
            hue = 0.67 * (1.0 - norm[i, j])
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            rgb[i, j] = [int(r*255), int(g*255), int(b*255)]
    return rgb


def draw_annotations(img_pil, annotations, color=(255, 80, 0), label="shrub"):
    """Draw existing shrub polygon annotations in orange."""
    draw = ImageDraw.Draw(img_pil, "RGBA")
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        for seg in ann.get("segmentation", []):
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg)-1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, outline=color + (220,), fill=color + (40,))
    return img_pil


def export_patch(d, out_rgb_dir, out_chm_dir, scale=4):
    """
    Export one patch as RGB PNG and CHM heatmap PNG.
    scale: upscale factor for visibility (128px → 512px default).
    Returns the relative path to the RGB PNG.
    """
    img5 = np.load(d["file_name"]).astype(np.float32)
    img5 = np.nan_to_num(img5, nan=0.0)

    # RGB PNG
    r = stretch(img5[0])
    g = stretch(img5[1])
    b = stretch(img5[2])
    rgb = np.stack([r, g, b], axis=2)  # (H, W, 3)
    rgb_pil = Image.fromarray(rgb).resize(
        (rgb.shape[1]*scale, rgb.shape[0]*scale), Image.NEAREST
    )
    # Draw existing shrub annotations (orange outlines)
    if d.get("annotations"):
        # Scale annotations for the upscaled image
        scaled_anns = []
        for ann in d["annotations"]:
            scaled_seg = []
            for seg in ann.get("segmentation", []):
                scaled_seg.append([c * scale for c in seg])
            scaled_anns.append({**ann, "segmentation": scaled_seg})
        rgb_pil = draw_annotations(rgb_pil, scaled_anns)

    stem = Path(d["file_name"]).stem
    rgb_path = out_rgb_dir / f"{stem}.png"
    rgb_pil.save(rgb_path)

    # CHM heatmap PNG
    chm = img5[CHM_CHANNEL]
    chm_rgb = chm_to_colormap(chm)
    chm_pil = Image.fromarray(chm_rgb).resize(
        (chm_rgb.shape[1]*scale, chm_rgb.shape[0]*scale), Image.NEAREST
    )
    chm_path = out_chm_dir / f"{stem}_chm.png"
    chm_pil.save(chm_path)

    return str(rgb_path), str(chm_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(TRAIN_JSON) as f:
        train = json.load(f)
    with open(VAL_JSON) as f:
        val = json.load(f)
    all_dicts = train + val

    sites = sorted(set(d["site"] for d in all_dicts))
    print(f"Sites: {sites}")
    print(f"Total patches: {len(all_dicts)}")

    OUT_DIR.mkdir(exist_ok=True)

    label_studio_tasks = []
    total = 0

    for site in sites:
        site_dicts = [d for d in all_dicts if d["site"] == site]
        annotated  = sum(1 for d in site_dicts if d.get("annotations"))
        empty      = len(site_dicts) - annotated

        out_rgb = OUT_DIR / site
        out_chm = OUT_DIR / f"{site}_chm"
        out_rgb.mkdir(exist_ok=True)
        out_chm.mkdir(exist_ok=True)

        print(f"\n[{site}] {len(site_dicts)} patches ({annotated} annotated, {empty} empty)")

        for i, d in enumerate(site_dicts):
            rgb_path, chm_path = export_patch(d, out_rgb, out_chm)
            total += 1

            # Label Studio task entry
            task = {
                "data": {
                    "image":    f"/data/local-files/?d={rgb_path}",
                    "chm":      f"/data/local-files/?d={chm_path}",
                    "site":     site,
                    "split":    "train" if d in train else "val",
                    "has_shrub_gt": bool(d.get("annotations")),
                    "patch_id": d["image_id"],
                    "source_mask": d.get("source_mask", ""),
                },
                "meta": {
                    "file_name": d["file_name"],
                }
            }

            # Pre-annotate existing shrub polygons as Label Studio predictions
            if d.get("annotations"):
                results = []
                for ann in d["annotations"]:
                    for seg in ann.get("segmentation", []):
                        pts = [{"x": seg[i]/d["width"]*100,
                                "y": seg[i+1]/d["height"]*100}
                               for i in range(0, len(seg)-1, 2)]
                        if len(pts) >= 3:
                            results.append({
                                "type": "polygonlabels",
                                "value": {
                                    "points": [[p["x"], p["y"]] for p in pts],
                                    "polygonlabels": ["shrub"],
                                },
                                "origin": "prediction",
                                "score": 1.0,
                            })
                if results:
                    task["predictions"] = [{"result": results, "score": 1.0}]

            label_studio_tasks.append(task)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(site_dicts)} exported …")

        print(f"  Done — saved to {out_rgb}/")

    # Save Label Studio import JSON
    ls_json_path = OUT_DIR / "label_studio_import.json"
    with open(ls_json_path, "w") as f:
        json.dump(label_studio_tasks, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exported {total} patches across {len(sites)} sites")
    print(f"Label Studio import file: {ls_json_path}")
    print(f"\nLabel Studio setup:")
    print(f"  1. pip install label-studio")
    print(f"  2. label-studio start")
    print(f"  3. Create project → Labeling Interface → Code tab → paste config below")
    print(f"  4. Import → {ls_json_path}")
    print()
    print("Label Studio XML config:")
    print("""
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Header value="CHM (height): blue=0m green=2m red=4m+"/>
  <Image name="chm" value="$chm" zoom="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2" pointSize="small">
    <Label value="shrub"  background="#FF6600"/>
    <Label value="tree"   background="#006600"/>
    <Label value="rock"   background="#888888"/>
    <Label value="shadow" background="#000066"/>
    <Label value="other_fp" background="#CC0000"/>
  </PolygonLabels>
</View>
""")


if __name__ == "__main__":
    main()

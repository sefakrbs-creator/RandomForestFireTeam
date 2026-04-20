# Shrubwise Dataset — CLAUDE.md

## Project Overview

This is a **shrub detection pipeline** that produces georeferenced shrub probability maps for aerial (NAIP) imagery across 6 California field sites. Ground truth labels come from SAM polygon annotations on 5-channel (NAIP + CHM) patches derived from Terrestrial LiDAR Scans (TLS).

## Repository Structure

```
Dataset/
├── CLAUDE.md
├── process_shrub_masks.py           # Label generation: TLS → NAIP mask rasters
├── shrub_mask_pipeline.ipynb        # Full notebook pipeline (downloads from remote WebDAV)
│
├── export_patches_for_labeling.py   # Export patches as PNGs + Label Studio import JSON (pre-SAM review)
├── sam_annotate.py                  # SAM refinement: circular → precise masks + hard negatives
├── train_shrub_v12.py               # Train XGBoost pixel classifier (V12)
├── predict_raster_v12.py            # V12 inference module (single AOI, called by agent)
├── predict_v12_all_sites.py         # Run V12 inference on all 6 known sites → prob + binary mask rasters
│
├── Model_Prediction/
│   ├── autonomous_shrub_agent.py    # End-to-end agent: download NAIP+CHM → V12 inference
│   ├── naip.tif                     # Downloaded NAIP (cached)
│   ├── chm.tif                      # Downloaded CHM (cached)
│   └── predicted_shrub_v12.tif      # Latest binary prediction output
│
├── shrub_classifier_v12.joblib      # Trained V12 XGBoost model
├── shrub_scaler_v12.joblib          # RobustScaler for V12 features (19 features, excl. canopy_in_shrub_range)
├── v12_model_meta.json              # V12 feature list, threshold, metrics
│
├── detectron2_dataset/              # Tiled .npy patches + SAM COCO JSON annotations
│   ├── shrub_train_sam.json
│   ├── shrub_val_sam.json
│   └── *.npy                        # 128×128 px, 5-channel (NAIP RGBNIR + CHM)
│
├── generate_all_sam_viz.py          # Generate PNG visualizations for ALL SAM tiles (both splits)
├── sam_viz_all/                     # Output: per-tile PNGs + per-site grids + summary_grid.png
│   ├── Calaveras_Big_trees/         # 463 individual tile PNGs
│   ├── DL_Bliss/                    # 40 individual tile PNGs
│   ├── Independence_Lake/           # 132 individual tile PNGs
│   ├── Pacific_Union/               # 53 individual tile PNGs
│   ├── Sedgwick/                    # 151 individual tile PNGs
│   ├── Shaver_Lake/                 # 40 individual tile PNGs
│   ├── *_grid.png                   # Per-site overview grids (6-column layout)
│   └── summary_grid.png             # Cross-site summary (4 patches per site)
│
├── Calaveras_Big_trees/
├── DL_Bliss/
├── Independence_Lake/
├── Pacific_Union/
├── Sedgwick/
└── Shaver_Lake/
```

### Per-site folder layout

```
<Site>/
├── aligned_TLS/           # Aligned TLS .laz files (one per plot scan)
├── ALS/                   # Airborne LiDAR tiles (.laz, USGS LPC)
├── transformations/       # 4x4 rigid-body transform matrices (.txt, one per shrub list)
├── shrub_lists_revised/   # Corrected shrub CSVs (preferred over shrub_lists/)
├── NAIP_3DEP_product/     # NAIP + CHM rasters
└── mask_outputs_sprint4/  # Pipeline outputs: masks, multiband TIFs, GeoJSONs
```

## Sites

| Site | Code | Notes |
|---|---|---|
| Calaveras Big Trees | CATCU | Multiple scan dates (May, Jun 2025, Oct 2024) |
| DL Bliss | CAAEU | Most complete — has revised shrub lists and sprint4 outputs |
| Independence Lake | — | Many per-polygon TIFs already exported (poly4–49) |
| Pacific Union College | — | Nov 2024 scan |
| Sedgwick | — | Multiple dates (Sep 2024, Nov 2024, Jun 2025, Jul 2025) |
| Shaver Lake | — | Jul 2024 scan |

## Pipeline Step Order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1 │ process_shrub_masks.py                                            │
│          TLS → NAIP Mask Rasters                                            │
│  IN:  TLS .laz + NAIP rasters                                               │
│  OUT: shrub mask rasters                                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2 │ Patch Extraction                                                  │
│          128×128 px Patch Tiling                                            │
│  IN:  NAIP + CHM rasters                                                    │
│  OUT: shrub_train.json / shrub_val.json + *.npy                             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3 │ export_patches_for_labeling.py                                    │
│          Label Studio Export                                                │
│  IN:  shrub_train.json / shrub_val.json                                     │
│  OUT: label_studio_images/ + label_studio_import.json                       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4 │ sam_annotate.py                                                   │
│          SAM Annotation Refinement                                          │
│  IN:  shrub_train.json + SAM checkpoint (sam_vit_h_4b8939.pth)              │
│  OUT: shrub_train_sam.json / shrub_val_sam.json                             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5 │ train_shrub_v12.py                                                │
│          XGBoost Training (V12)                                             │
│  IN:  shrub_train_sam.json + *.npy patches                                  │
│  OUT: shrub_classifier_v12.joblib / shrub_scaler_v12.joblib /               │
│       v12_model_meta.json                                                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6 │ predict_v12_all_sites.py                                          │
│          V12 Inference — All Sites                                          │
│  IN:  NAIP + CHM rasters (6 sites)                                          │
│  OUT: predictions_v12/<site>/*_v12_prob.tif / *_v12_mask.tif               │
└────────────────────┬───────────────────────────────┬────────────────────────┘
                     │                               │
           Known sites (batch)            New AOI (any location)
                     │                               │
                     ▼                               ▼
        predictions_v12/<site>/   ┌─────────────────────────────────────────┐
        *_v12_prob.tif            │  STEP 7 │ autonomous_shrub_agent.py     │
        *_v12_mask.tif            │  🤖 AI AGENT — Predict New Sites        │
                                  │  IN:  GeoJSON AOI file                  │
                                  │  AUTO: downloads NAIP from Planetary    │
                                  │        Computer + CHM from GEE          │
                                  │  OUT: naip.tif / chm.tif /             │
                                  │       predicted_shrub_v12.tif           │
                                  │       predicted_shrub_v12_prob.tif      │
                                  └─────────────────────────────────────────┘
```

> Visual diagram also rendered at runtime in `shrub_pipeline.ipynb` (first cell)
> and saved to `predictions_v12/pipeline_diagram.png`.

---

## Label Studio Export — export_patches_for_labeling.py

Exports all dataset patches as PNGs for manual review in Label Studio **before SAM refinement**. This lets you visually verify and correct the initial circular shrub annotations.

### Outputs
| Path | Description |
|---|---|
| `label_studio_images/<site>/` | RGB PNGs (NAIP R, G, B), 512×512 px (4× upscaled) |
| `label_studio_images/<site>_chm/` | CHM false-color heatmaps (blue=0m → green=2m → red≥4m) |
| `label_studio_images/label_studio_import.json` | Pre-annotated import file with existing shrub polygons |

### What it does
- Loads `shrub_train.json` + `shrub_val.json` (raw circular annotations)
- Upscales each 128×128 px patch to 512×512 (4× NEAREST) for visibility
- Draws existing shrub polygon annotations in orange on the RGB PNGs
- Generates `label_studio_import.json` with all patches pre-annotated for Label Studio

### Label Studio setup
```bash
pip install label-studio
label-studio start
# Then: Create project → Import → label_studio_images/label_studio_import.json
```

Label config (5 classes): `shrub` (orange), `tree` (green), `rock` (grey), `shadow` (dark blue), `other_fp` (red)

### Usage
```bash
python export_patches_for_labeling.py
```

---

## SAM Annotation Pipeline — sam_annotate.py

Refines raw circular shrub annotations and mines hard negatives using Segment Anything Model (SAM). Runs in two passes per patch.

### Pass 1 — Prompted SAM (shrub refinement)
For each existing circular shrub annotation:
1. Use mask centroid as a foreground point prompt
2. SAM predicts a precise mask fitting the actual shrub shape
3. Replace circular polygon with SAM mask polygon
4. Validate: keep only if mean CHM ∈ [1.0, 4.0] m — otherwise keep original

### Pass 2 — Automatic Mask Generation (hard negative mining)
Runs SAM AMG on every patch (annotated and empty):
- CHM > 4.0 m → label **tree** (category_id=1)
- CHM < 1.0 m → label **rock** (category_id=2)
- IoU > 0.3 with any existing shrub mask → skip
- Area < 4 px² or > 3000 px² → skip (noise / background)

### Key constants
| Constant | Value | Description |
|---|---|---|
| `UPSCALE` | 4 | Patches upscaled 128→512 px before SAM for better small-object detection |
| `CHM_SHRUB_LO/HI` | 1.0 / 4.0 m | Valid shrub CHM range |
| `CHM_TREE_THR` | 4.0 m | Above → tree hard negative |
| `CHM_ROCK_THR` | 1.0 m | Below → rock hard negative |
| `AMG_POINTS_PER_SIDE` | 32 | Dense grid for small shrubs |
| `AMG_PRED_IOU_THRESH` | 0.7 | SAM AMG quality filter |
| `AMG_STABILITY_THRESH` | 0.85 | SAM AMG stability filter |
| `IOU_OVERLAP` | 0.3 | Hard negative suppressed if IoU with shrub exceeds this |

### Inputs / Outputs
| File | Description |
|---|---|
| `detectron2_dataset/shrub_train.json` | Input — raw circular annotations (train) |
| `detectron2_dataset/shrub_val.json` | Input — raw circular annotations (val) |
| `detectron2_dataset/shrub_train_sam.json` | Output — SAM-refined train annotations |
| `detectron2_dataset/shrub_val_sam.json` | Output — SAM-refined val annotations |
| `detectron2_dataset/sam_stats.json` | Output — per-split annotation counts |

### Usage (WSL, venv_linux activated)
```bash
# Default (SAM ViT-H)
python sam_annotate.py

# Faster ViT-B model
python sam_annotate.py --model-type vit_b --checkpoint sam_vit_b_01ec64.pth

# Single site only
python sam_annotate.py --sites DL_Bliss

# Shrub refinement only, skip hard negative AMG (much faster)
python sam_annotate.py --skip-amg
```

### SAM checkpoint required
Default: `./sam_vit_h_4b8939.pth` (ViT-H, ~2.5 GB). Must be downloaded separately from Meta's SAM release.

---

## SAM Annotations

Labels come from SAM polygon annotations stored in COCO instance format:
- `detectron2_dataset/shrub_train_sam.json` — training set
- `detectron2_dataset/shrub_val_sam.json` — validation set

Each record references a 128×128 px `.npy` patch (5 channels: NAIP R, G, B, NIR + CHM).

**Label scheme (binary: shrub vs non-shrub):**
| category_id | Class | Label |
|---|---|---|
| 0 | shrub | 1 (positive) |
| 1 | tree | 0 (hard negative, CHM > 4 m) |
| 2 | rock | 0 (hard negative, CHM < 1 m) |
| — | unlabeled pixels | skipped (not used) |

### SAM Tile Visualizations — generate_all_sam_viz.py

PNG visualizations for all 879 SAM-annotated patches generated via `generate_all_sam_viz.py`.

**Tile counts per site (train + val combined):**
```
┌─────────────────────┬───────┐
│        Site         │ Tiles │
├─────────────────────┼───────┤
│ Calaveras_Big_trees │ 463   │
├─────────────────────┼───────┤
│ DL_Bliss            │ 40    │
├─────────────────────┼───────┤
│ Independence_Lake   │ 132   │
├─────────────────────┼───────┤
│ Pacific_Union       │ 53    │
├─────────────────────┼───────┤
│ Sedgwick            │ 151   │
├─────────────────────┼───────┤
│ Shaver_Lake         │ 40    │
├─────────────────────┼───────┤
│ Total               │ 879   │
└─────────────────────┴───────┘
```

**Outputs in `sam_viz_all/`:**
| File | Description |
|---|---|
| `<site>/*.png` | Individual tile PNG (RGB + annotation overlays) |
| `<site>_grid.png` | All tiles for that site in a 6-column grid |
| `summary_grid.png` | Cross-site overview — 4 patches per site |

**Color legend:** green = shrub, red = tree (hard-neg), blue = rock (hard-neg). Thicker edges = SAM-refined.

**Run:**
```bash
python generate_all_sam_viz.py
# Options: --out-dir sam_viz_all --splits train val --dpi 100 --no-individual
```

Generated 2026-04-19 (156 s total, dpi=100).

## Model — XGBoost Pixel Classifier (V12)

**Script:** `train_shrub_v12.py`  
**Saved artifacts:** `shrub_classifier_v12.joblib`, `shrub_scaler_v12.joblib`, `v12_model_meta.json`

A binary per-pixel classifier trained directly from SAM polygon labels rasterized onto the .npy patches.

### Training pipeline
1. **Rasterize SAM polygons** → per-pixel label map (-1=unlabeled, 0=non-shrub, 1=shrub)
2. **Extract 55 features** per labeled pixel (NAIP + CHM + spectral indices + multi-scale texture)
3. **Subsample negatives** to MAX_NEG_PER_PATCH=200 per patch (prevents OOM)
4. **Pass 1** — fit XGBoost on all 55 features, rank by importance, select top 20
5. **Pass 2** — refit RobustScaler on top-20 features (excluding `canopy_in_shrub_range`); 80-trial Optuna NSGA-II search maximising F1/Precision/Recall; Pareto-front best selected by F1
6. **Final model** — retrain with best params, n_estimators=2000, early_stopping_rounds=50

### Input features — top 20 selected (from `v12_model_meta.json`)

Selected by Pass-1 XGBoost importance out of 55 candidates. Sorted by importance score.

| Rank | Feature | Group | Importance |
|---|---|---|---|
| 1 | `canopy_height_7x7mean` | Canopy height texture | 0.1580 |
| 2 | `canopy_height_5x5mean` | Canopy height texture | 0.0931 |
| 3 | `naip_red_5x5std` | NAIP texture | 0.0786 |
| 4 | `brightness_7x7std` | Spectral index texture | 0.0659 |
| 5 | `canopy_height_3x3mean` | Canopy height texture | 0.0428 |
| 6 | `naip_red_3x3std` | NAIP texture | 0.0279 |
| 7 | `brightness_5x5std` | Spectral index texture | 0.0247 |
| 8 | `canopy_height_7x7std` | Canopy height texture | 0.0234 |
| 9 | `canopy_shrub_clipped` | Canopy height derived | 0.0228 |
| 10 | `savi_7x7std` | Spectral index texture | 0.0221 |
| 11 | `canopy_in_shrub_range` | Canopy height derived (binary, not scaled) | 0.0185 |
| 12 | `naip_red_7x7std` | NAIP texture | 0.0184 |
| 13 | `greenness_ratio` | Spectral index | 0.0180 |
| 14 | `brightness_3x3mean` | Spectral index texture | 0.0173 |
| 15 | `savi_7x7mean` | Spectral index texture | 0.0172 |
| 16 | `naip_nir_7x7std` | NAIP texture | 0.0169 |
| 17 | `naip_green_3x3std` | NAIP texture | 0.0160 |
| 18 | `ndvi_7x7mean` | Spectral index texture | 0.0155 |
| 19 | `canopy_height_5x5std` | Canopy height texture | 0.0154 |
| 20 | `gndvi` | Spectral index | 0.0144 |

**Key finding:** Canopy height features dominate (ranks 1, 2, 5, 8, 19 = ~48% cumulative importance). Multi-scale texture (mean/std at 3×3, 5×5, 7×7 windows) consistently outperforms raw spectral values.

> **Scaling note:** `canopy_in_shrub_range` is a binary 0/1 flag listed in `no_scale_features` — it must NOT be passed through RobustScaler. The scaler was fit on the other 19 features only. Always check `v12_model_meta.json` → `"no_scale_features"` before applying the scaler.

### Metrics (val set)
| Metric | Value |
|---|---|
| AUC | 0.7723 |
| Precision | 0.7533 |
| Recall | 0.9065 |
| F1 | 0.8229 |
| Threshold | 0.5443 |

### Best hyperparameters (Optuna NSGA-II, 80 trials)
| Parameter | Value |
|---|---|
| `learning_rate` | 0.1136 |
| `max_depth` | 8 |
| `min_child_weight` | 2.99 |
| `gamma` | 4.77 |
| `subsample` | 0.869 |
| `colsample_bytree` | 0.688 |
| `reg_alpha` | 0.0032 |
| `reg_lambda` | 5.98e-05 |
| `scale_pos_weight` | 8.18 |
| `n_estimators` | 2000 |

### Required inputs for inference
- NAIP 4-band TIF (R, G, B, NIR) — `<Site>/NAIP_3DEP_product/<prefix>_1m_naip_2022.tif`
- Canopy height TIF (CHM) — `<Site>/NAIP_3DEP_product/<prefix>_canopy_height_1m.tif`
- **No DEM/terrain required**

### Confirmed input files per site
| Site | NAIP | CHM |
|---|---|---|
| Calaveras_Big_trees | `calaveras_big_trees_1m_naip_2022.tif` | `calaveras_big_trees_canopy_height_1m.tif` |
| DL_Bliss | `dl_bliss_1m_naip_2022.tif` | `dl_bliss_canopy_height_1m.tif` |
| Independence_Lake | `independence_lake_1m_naip_2022.tif` | `independence_lake_canopy_height_1m.tif` |
| Pacific_Union | `pacific_union_college_1m_naip_2022.tif` | `pacific_union_canopy_height_1m.tif` |
| Sedgwick | `sedgwick_1m_naip_2022.tif` | `sedgwick_canopy_height_1m.tif` |
| Shaver_Lake | `shaver_lake_1m_naip_2022.tif` | `shaver_lake_canopy_height_1m.tif` |

All inputs confirmed present (verified 2026-04-18). CRS: EPSG:6350 (NAD83(2011) / Conus Albers).

## Required Packages

All scripts in this repo depend on the following packages. Install into `venv_linux` under WSL.

### Core data & ML
```
numpy                 # Array operations throughout
pandas                # DataFrames, CSV I/O
scipy                 # uniform_filter (texture features), cKDTree (LiDAR queries)
scikit-learn          # RobustScaler, metrics (precision/recall/F1/AUC)
xgboost               # V12 XGBoost classifier
optuna                # Hyperparameter search — NSGA-II multi-objective sampler
joblib                # Model serialization (joblib.dump / joblib.load)
```

### Geospatial
```
rasterio              # GeoTIFF read/write, CRS handling, rasterize, shapes, warp
geopandas             # GeoDataFrame, GeoJSON I/O, CRS reprojection
shapely               # Polygon / Point geometry, unary_union
```

### Remote sensing & data download
```
earthengine-api       # Google Earth Engine (import ee) — 3DEP DEM + CHM download
planetary-computer    # Microsoft Planetary Computer STAC authentication
pystac-client         # STAC API client (pystac_client.Client) — NAIP search & download
```

### Computer vision & annotation
```
opencv-python         # cv2 — polygon rasterization (fillPoly), contour extraction, image resize
segment-anything      # SAM — SamPredictor, SamAutomaticMaskGenerator (sam_annotate.py)
torch                 # PyTorch — required by SAM
torchvision           # NMS ops (used in old fusion scripts, still imported)
Pillow                # PIL.Image — image drawing utilities
```

### Visualisation
```
matplotlib            # Plotting probability maps, annotation overlays
```

### Install command (WSL, inside venv_linux)
```bash
pip install numpy pandas scipy scikit-learn xgboost optuna joblib \
            rasterio geopandas shapely \
            earthengine-api planetary-computer pystac-client \
            opencv-python torch torchvision Pillow matplotlib

# SAM — install from Meta's GitHub repo
pip install git+https://github.com/facebookresearch/segment-anything.git
# Then download checkpoint: sam_vit_h_4b8939.pth (~2.5 GB)
```

### Runtime environment
```
venv_linux/    # Linux venv — must run under WSL (Ubuntu)
               # Activate: source venv_linux/bin/activate
```

## Inference — predict_v12_all_sites.py

Runs V12 on full-site NAIP + CHM rasters. Outputs per site saved to `predictions_v12/<Site>/`:
- `<site>_v12_prob.tif` — float32 shrub probability map (0–1)
- `<site>_v12_mask.tif` — uint8 binary shrub mask at threshold

```bash
# All sites
python predict_v12_all_sites.py

# Single site
python predict_v12_all_sites.py --sites DL_Bliss

# Custom threshold (default: 0.544 from meta)
python predict_v12_all_sites.py --threshold 0.30
```

### Site results (2026-04-19, threshold=0.544)
| Site | Raster size | Shrub ha | p50 | p95 |
|---|---|---|---|---|
| Calaveras_Big_trees | 1537×4107 | 182.5 | 0.394 | 0.851 |
| DL_Bliss | 1124×578 | 40.8 | 0.667 | 0.915 |
| Independence_Lake | 1161×1760 | 151.3 | 0.733 | 0.921 |
| Pacific_Union | 2716×1087 | 219.1 | 0.735 | 0.889 |
| Sedgwick | 1815×589 | 103.9 | 0.872 | 0.945 |
| Shaver_Lake | 1078×531 | 35.6 | 0.655 | 0.909 |

Total shrub area: ~733 ha across 6 sites.

## Autonomous Agent — Model_Prediction/autonomous_shrub_agent.py

Downloads required data layers for any AOI and runs V12 inference end-to-end.

### Pipeline steps
1. Initialize Google Earth Engine (`ee-sefakarabas` project)
2. Load GeoJSON AOI → reproject to EPSG:6350 if geographic
3. Download NAIP (4-band, 1 m/px) from Planetary Computer if missing
4. Download CHM (Meta Canopy Height) from GEE aligned to NAIP grid if missing
5. Run `predict_raster_v12.run_shrub_prediction_v12()` → binary mask + prob raster

**No DEM download** — V12 requires only NAIP + CHM.

### Outputs (in `--output` directory)
| File | Description |
|---|---|
| `naip.tif` | Downloaded NAIP 4-band raster |
| `chm.tif` | Downloaded canopy height raster |
| `predicted_shrub_v12.tif` | Binary shrub mask (uint8, threshold=0.544) |
| `predicted_shrub_v12_prob.tif` | Float32 shrub probability map (0–1) |

### Usage
```bash
python Model_Prediction/autonomous_shrub_agent.py my_aoi.geojson
python Model_Prediction/autonomous_shrub_agent.py my_aoi.geojson --output results/ --year 2022
```

### Supporting module — predict_raster_v12.py
Standalone inference module (sibling of `predict_raster_v10.py`) with a single entry point:
```python
from predict_raster_v12 import run_shrub_prediction_v12
run_shrub_prediction_v12(naip_path, chm_path, output_path)
```
Model paths resolve relative to the script file — works from any working directory.

---

## Jupyter Pipeline Notebook — shrub_pipeline.ipynb

End-to-end pipeline notebook with 8 sections. Runs headlessly via `jupyter nbconvert --execute`.

### Sections
| # | Section | Description |
|---|---|---|
| 1 | Logging Setup | Creates `logs/{RUN_TS}_pipeline.log`; sets up `log()` and `log_section()` helpers |
| 2 | Environment Check | Imports all required packages, reports versions and GPU info |
| 3 | Artifact & Input File Check | Verifies model files and per-site NAIP/CHM rasters |
| 4 | SAM Annotation Summary | Reads `sam_stats.json`, reports shrub/tree/rock counts per split |
| 5 | V12 Training | Runs `train_shrub_v12.py` as subprocess (skipped if model exists and `FORCE_RETRAIN=False`) |
| 6 | V12 Prediction | Runs all 6 sites in-process; logs per-site results and summary table |
| 7 | Visualisation | Saves `predictions_v12/all_sites_overview.png` and `probability_histograms.png` |
| 8 | View Run Log | Flushes handlers, prints full log, lists all historical runs in `logs/` |

### Logging
- Log files: `logs/{RUN_TS}_pipeline.log` (timestamped per run)
- Training subprocess output: `logs/{RUN_TS}_training.log`
- Dual output: file handler + stdout (visible in notebook cells)
- `FORCE_RETRAIN = False` flag at top of cell 5 — set `True` to force retraining

### Test run (headless)
```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
  shrub_pipeline.ipynb --output shrub_pipeline_executed.ipynb
```

Test executed successfully 2026-04-19 (all 6 sites, ~55s total, exit code 0).

---

## Running from Claude Code (Windows → WSL)

```
! wsl bash -c "cd /mnt/c/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2 && source venv_linux/bin/activate && PYTHONUNBUFFERED=1 python train_shrub_v12.py 2>&1"
```

```
! wsl bash -c "cd /mnt/c/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2 && source venv_linux/bin/activate && PYTHONUNBUFFERED=1 python predict_v12_all_sites.py 2>&1"
```

"""
train_shrub_v12.py
==================
Binary shrub pixel classifier — uses SAM annotations directly.

Key difference from v11:
  Instead of deriving labels from CHM thresholds on the CSV,
  this script rasterizes the actual SAM polygon labels from
  shrub_train_sam.json / shrub_val_sam.json onto the .npy patches
  to produce precise per-pixel ground truth.

  Label scheme (binary: shrub vs non-shrub):
    category_id=0 (shrub)  → y=1  (positives, SAM-refined boundaries)
    category_id=1 (tree)   → y=0  (hard negatives, CHM>4m confirmed by SAM AMG)
    category_id=2 (rock)   → y=0  (hard negatives, CHM<1m confirmed by SAM AMG)
    unlabeled pixels        → skipped (not used in training)

  Features per pixel (from .npy patch):
    NAIP: R, G, B, NIR
    CHM (canopy height)
    Derived: NDVI, SAVI, GNDVI, EVI2, brightness, greenness_ratio
    Canopy-derived: canopy_in_shrub_range, canopy_shrub_clipped
    Local texture (3×3, 5×5, 7×7 mean/std within patch)

  Same training pipeline as v11:
    Two-pass XGBoost feature selection → Optuna NSGA-II multi-objective search
    Early stopping in every XGBoost fit

Output: shrub_classifier_v12.joblib / shrub_scaler_v12.joblib / v12_model_meta.json
"""

import json
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from scipy.ndimage import uniform_filter
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve,
)
from sklearn.preprocessing import RobustScaler
import optuna
import joblib
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_JSON = "./detectron2_dataset/shrub_train_sam.json"
VAL_JSON   = "./detectron2_dataset/shrub_val_sam.json"

CHM_CHANNEL  = 4
NIR_CHANNEL  = 3
RED_CHANNEL  = 0
SHRUB_MIN_H  = 1.0
SHRUB_MAX_H  = 4.0

TOP_K_FEATURES    = 20
NO_SCALE          = {"canopy_in_shrub_range"}
EARLY_STOP_ROUNDS = 30
N_TRIALS          = 80


# ---------------------------------------------------------------------------
# 1. RASTERIZE SAM POLYGONS → PER-PIXEL FEATURE ROWS
# ---------------------------------------------------------------------------

def poly_to_mask(segmentation, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def extract_patch_features(img5, H, W):
    """
    img5: (5, H, W) float32  — NAIP RGBNIR + CHM
    Returns a dict of 2-D arrays (H, W) — one per feature.
    """
    R   = img5[RED_CHANNEL].astype(np.float32)
    G   = img5[1].astype(np.float32)
    B   = img5[2].astype(np.float32)
    NIR = img5[NIR_CHANNEL].astype(np.float32)
    CHM = img5[CHM_CHANNEL].astype(np.float32)

    feats = {
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
        "canopy_in_shrub_range": ((CHM >= SHRUB_MIN_H) & (CHM <= SHRUB_MAX_H)).astype(np.float32),
        "canopy_shrub_clipped":  np.clip(CHM, 0, SHRUB_MAX_H),
    }

    # Multi-scale texture on key bands
    texture_sources = ["ndvi", "naip_nir", "naip_red", "naip_green",
                       "savi", "brightness", "canopy_height"]
    for src in texture_sources:
        arr = feats[src]
        for win in [3, 5, 7]:
            mean_val = uniform_filter(arr, size=win, mode="reflect")
            sq_mean  = uniform_filter(arr ** 2, size=win, mode="reflect")
            std_val  = np.sqrt(np.clip(sq_mean - mean_val ** 2, 0, None))
            feats[f"{src}_{win}x{win}mean"] = mean_val
            feats[f"{src}_{win}x{win}std"]  = std_val

    return feats


MAX_NEG_PER_PATCH = 200   # cap hard-negative pixels per patch to control memory

def build_pixel_rows(json_path, split_name, rng=None):
    """
    Parse SAM JSON, rasterize polygons, extract features per labeled pixel.
    Uses numpy arrays throughout to avoid dict-per-row memory explosion.
    Returns (X_df, y_series, site_series).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    with open(json_path) as f:
        dicts = json.load(f)

    feat_names = None
    X_chunks, y_chunks, site_chunks = [], [], []

    print(f"  [{split_name}] {len(dicts)} patches — rasterizing SAM annotations...")
    t0 = time.time()

    for d in dicts:
        anns = d.get("annotations", [])
        if not anns:
            continue

        img5 = np.load(d["file_name"]).astype(np.float32)
        img5 = np.nan_to_num(img5, nan=0.0, posinf=0.0, neginf=0.0)
        _, H, W = img5.shape

        feats = extract_patch_features(img5, H, W)
        if feat_names is None:
            feat_names = list(feats.keys())

        # Build label map: -1=unlabeled, 0=non-shrub, 1=shrub
        label_map = np.full((H, W), -1, dtype=np.int8)
        for cat_id in [1, 2, 0]:   # shrub last — overwrites tree/rock if overlap
            for ann in anns:
                if ann.get("iscrowd", 0) or ann.get("category_id", 0) != cat_id:
                    continue
                m = poly_to_mask(ann["segmentation"], H, W)
                label_map[m] = 1 if cat_id == 0 else 0

        # Separate positives and negatives
        pos_idx = np.argwhere(label_map == 1)
        neg_idx = np.argwhere(label_map == 0)
        if len(pos_idx) == 0 and len(neg_idx) == 0:
            continue

        # Subsample negatives to cap memory usage
        if len(neg_idx) > MAX_NEG_PER_PATCH:
            neg_idx = neg_idx[rng.choice(len(neg_idx), MAX_NEG_PER_PATCH, replace=False)]

        idx = np.concatenate([pos_idx, neg_idx], axis=0)  # (N, 2)
        ys, xs = idx[:, 0], idx[:, 1]
        labels = label_map[ys, xs].astype(np.int8)

        feat_matrix = np.stack([feats[f][ys, xs] for f in feat_names], axis=1).astype(np.float32)
        X_chunks.append(feat_matrix)
        y_chunks.append(labels)
        site_chunks.append(np.full(len(labels), d.get("site", "unknown"), dtype=object))

    X_arr = np.concatenate(X_chunks, axis=0)
    y_arr = np.concatenate(y_chunks, axis=0)
    s_arr = np.concatenate(site_chunks, axis=0)

    print(f"  [{split_name}] Done in {time.time()-t0:.0f}s  — "
          f"{len(y_arr):,} pixels  pos={y_arr.sum():,} ({y_arr.mean()*100:.1f}%)")

    X_df = pd.DataFrame(X_arr, columns=feat_names)
    return X_df, pd.Series(y_arr, dtype=int), pd.Series(s_arr)


# ---------------------------------------------------------------------------
# 2. BUILD DATASETS
# ---------------------------------------------------------------------------
print("="*60)
print("Building pixel datasets from SAM annotations")
print("="*60)

X_train_raw, y_train, sites_train = build_pixel_rows(TRAIN_JSON, "train")
X_val_raw,   y_val,   sites_val   = build_pixel_rows(VAL_JSON,   "val")

CANDIDATE_FEATURES = list(X_train_raw.columns)
print(f"\nFeature count: {len(CANDIDATE_FEATURES)}")
print(f"Train: {len(X_train_raw):,} pixels  |  pos={y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"Val:   {len(X_val_raw):,} pixels   |  pos={y_val.sum():,}   ({y_val.mean()*100:.1f}%)")

print("\nPer-site train pixel counts:")
for site, grp in pd.DataFrame({"y": y_train, "site": sites_train}).groupby("site"):
    print(f"  {site}: {len(grp):,} pixels  shrub={grp['y'].sum():,} ({grp['y'].mean()*100:.1f}%)")


# ---------------------------------------------------------------------------
# 3. SCALE
# ---------------------------------------------------------------------------
def scale(X_df, scaler=None):
    scale_cols = [c for c in X_df.columns if c not in NO_SCALE]
    X = X_df.copy().astype(np.float32)
    X[scale_cols] = X[scale_cols].fillna(0.0)
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(X[scale_cols].values)
    X[scale_cols] = scaler.transform(X[scale_cols].values)
    X[[c for c in X.columns if c in NO_SCALE]] = \
        X[[c for c in X.columns if c in NO_SCALE]].fillna(0.0)
    return X, scaler


# ---------------------------------------------------------------------------
# 4. PASS 1 — IMPORTANCE RANKING
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Pass 1 — XGBoost importance ranking on all features")
print(f"{'='*60}")

X_train_s1, _ = scale(X_train_raw)
X_val_s1, _   = scale(X_val_raw)

spw = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
print(f"  scale_pos_weight = {spw:.2f}")

p1_model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", learning_rate=0.05, max_depth=6,
    n_estimators=500, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw,
    early_stopping_rounds=30, random_state=42, verbosity=0,
)
t0 = time.time()
p1_model.fit(X_train_s1, y_train, eval_set=[(X_val_s1, y_val)], verbose=False)
print(f"  Pass-1 done in {time.time()-t0:.0f}s")

imp_df = pd.DataFrame({
    "feature":    CANDIDATE_FEATURES,
    "importance": p1_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n  Top 25 features:")
print(imp_df.head(25).to_string(index=False))

TOP_FEATURES = imp_df["feature"].head(TOP_K_FEATURES).tolist()
print(f"\n  Selected top {TOP_K_FEATURES}: {TOP_FEATURES}")


# ---------------------------------------------------------------------------
# 5. PASS 2 — OPTUNA ON TOP-K FEATURES
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Pass 2 — Optuna NSGA-II on top-{TOP_K_FEATURES} features")
print(f"{'='*60}")

X_train_top, scaler = scale(X_train_raw[TOP_FEATURES])
X_val_top,   _      = scale(X_val_raw[TOP_FEATURES],   scaler)

def objective(trial):
    threshold = trial.suggest_float("threshold", 0.05, 0.95)
    param = dict(
        objective             = "binary:logistic",
        eval_metric           = "logloss",
        tree_method           = "hist",
        learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth             = trial.suggest_int("max_depth", 3, 10),
        n_estimators          = 500,
        min_child_weight      = trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        gamma                 = trial.suggest_float("gamma", 0.0, 5.0),
        subsample             = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree      = trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha             = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda            = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        scale_pos_weight      = trial.suggest_float("scale_pos_weight", 1.0, 30.0),
        early_stopping_rounds = EARLY_STOP_ROUNDS,
        random_state          = 42,
        verbosity             = 0,
    )
    m = xgb.XGBClassifier(**param)
    m.fit(X_train_top, y_train, eval_set=[(X_val_top, y_val)], verbose=False)
    proba = m.predict_proba(X_val_top)[:, 1]
    preds = (proba >= threshold).astype(int)
    f1 = float(f1_score(y_val, preds, zero_division=0))
    p  = float(precision_score(y_val, preds, zero_division=0))
    r  = float(recall_score(y_val, preds, zero_division=0))
    trial.set_user_attr("f1", f1)
    trial.set_user_attr("precision", p)
    trial.set_user_attr("recall", r)
    return f1, p, r

print(f"  Running {N_TRIALS} trials (NSGA-II, objectives: F1 / Precision / Recall)...")
t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    directions=["maximize", "maximize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)
study.optimize(objective, n_trials=N_TRIALS)
print(f"  Completed in {time.time()-t0:.0f}s")

pareto = study.best_trials
print(f"  Pareto front: {len(pareto)} trials")
print("  Top 5 (F1 / Precision / Recall / threshold):")
for tr in sorted(pareto, key=lambda t: t.values[0], reverse=True)[:5]:
    print(f"    F1={tr.values[0]:.4f}  P={tr.values[1]:.4f}  "
          f"R={tr.values[2]:.4f}  thr={tr.params['threshold']:.4f}")

best_trial = max(pareto, key=lambda t: t.values[0])
best_params = best_trial.params.copy()
optuna_threshold = best_params.pop("threshold")
best_params.update(dict(
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", n_estimators=2000,
    early_stopping_rounds=50, random_state=42, verbosity=0,
))

print("\n  Training final model...")
cls_model = xgb.XGBClassifier(**best_params)
cls_model.fit(X_train_top, y_train, eval_set=[(X_val_top, y_val)], verbose=False)


# ---------------------------------------------------------------------------
# 6. THRESHOLD + EVALUATION
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Threshold & Evaluation")
print(f"{'='*60}")

val_proba = cls_model.predict_proba(X_val_top)[:, 1]
preds_opt = (val_proba >= optuna_threshold).astype(int)
print(f"  Optuna threshold: {optuna_threshold:.4f}")
print(f"  Val F1:           {f1_score(y_val, preds_opt, zero_division=0):.4f}")
print(f"  Val Precision:    {precision_score(y_val, preds_opt, zero_division=0):.4f}")
print(f"  Val Recall:       {recall_score(y_val, preds_opt, zero_division=0):.4f}")

prec_c, rec_c, thresholds = precision_recall_curve(y_val, val_proba)
f1_curve = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
pr_idx   = int(np.argmax(f1_curve))
pr_thr   = float(thresholds[pr_idx]) if pr_idx < len(thresholds) else 0.5
print(f"\n  (PR-curve F1-max: thr={pr_thr:.4f}  "
      f"F1={f1_curve[pr_idx]:.4f}  P={prec_c[pr_idx]:.4f}  R={rec_c[pr_idx]:.4f})")

# Val-set is the test set here (no separate test split since train/val come from SAM JSON)
auc  = roc_auc_score(y_val, val_proba)
prec = precision_score(y_val, preds_opt, zero_division=0)
rec  = recall_score(y_val, preds_opt, zero_division=0)
f1   = f1_score(y_val, preds_opt, zero_division=0)
print(f"\n  Final val metrics (threshold={optuna_threshold:.4f}):")
print(f"    AUC:       {auc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    F1:        {f1:.4f}")


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
print(f"\nTop {TOP_K_FEATURES} features (final model):")
final_imp = pd.DataFrame({
    "feature":    TOP_FEATURES,
    "importance": cls_model.feature_importances_,
}).sort_values("importance", ascending=False)
print(final_imp.to_string(index=False))


# ---------------------------------------------------------------------------
# 8. SAVE
# ---------------------------------------------------------------------------
joblib.dump(cls_model, "shrub_classifier_v12.joblib")
joblib.dump(scaler,    "shrub_scaler_v12.joblib")

meta = {
    "version": "v12",
    "inference_safe": True,
    "changes_from_v11": [
        "Labels from SAM polygon rasterization instead of CHM thresholds on CSV",
        "Training data: labeled pixels from SAM-annotated 128x128 patches",
        "Shrub (cat 0)=1, Tree (cat 1)=0, Rock (cat 2)=0, unlabeled=skipped",
        "No terrain features (elevation/slope/aspect) — not in .npy patches",
        "Texture computed per-patch (reflect padding instead of grid layout)",
    ],
    "data_source": {
        "train_json": TRAIN_JSON,
        "val_json":   VAL_JSON,
    },
    "feature_selection": {
        "candidate_count": len(CANDIDATE_FEATURES),
        "top_k": TOP_K_FEATURES,
        "pass1_importance_ranking": imp_df.to_dict(orient="records"),
    },
    "features": TOP_FEATURES,
    "scaled_features": [f for f in TOP_FEATURES if f not in NO_SCALE],
    "no_scale_features": sorted(NO_SCALE & set(TOP_FEATURES)),
    "cls_threshold": round(float(optuna_threshold), 4),
    "metrics": {
        "auc":       round(float(auc), 4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "f1":        round(float(f1), 4),
    },
    "best_params": {k: v for k, v in best_params.items()
                    if k not in ("verbosity", "random_state", "objective",
                                 "eval_metric", "tree_method",
                                 "early_stopping_rounds")},
}
with open("v12_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
print("  shrub_classifier_v12.joblib")
print("  shrub_scaler_v12.joblib")
print("  v12_model_meta.json")
print(f"  Threshold: {optuna_threshold:.4f}")
print("\nDone.")

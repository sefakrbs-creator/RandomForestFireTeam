"""
evaluate_v12.py
===============
Statistics and threshold-accuracy analysis for the V12 XGBoost shrub classifier.

Outputs (saved to output/v12_eval/):
  threshold_table.csv   — metrics at every swept threshold
  roc_curve.png
  pr_curve.png
  confusion_matrix.png  — at Optuna threshold
  feature_importance.png
  threshold_sweep.png   — F1 / Precision / Recall vs threshold

Run:
  python evaluate_v12.py
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.ndimage import uniform_filter
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    accuracy_score, jaccard_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE       = Path(__file__).parent
META_PATH  = BASE / "v12_model_meta.json"
MODEL_PATH = BASE / "shrub_classifier_v12.joblib"
SCALER_PATH= BASE / "shrub_scaler_v12.joblib"
VAL_JSON   = BASE / "detectron2_dataset/shrub_val_sam.json"
OUT_DIR    = BASE / "output/v12_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHM_CHANNEL  = 4
NIR_CHANNEL  = 3
RED_CHANNEL  = 0
SHRUB_MIN_H  = 1.0
SHRUB_MAX_H  = 4.0
MAX_NEG_PER_PATCH = 200

THRESHOLD_SWEEP = np.round(np.arange(0.05, 1.00, 0.05), 2)

# ---------------------------------------------------------------------------
# Feature extraction  (mirrors train_shrub_v12.py exactly)
# ---------------------------------------------------------------------------

def poly_to_mask(segmentation, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def extract_patch_features(img5, H, W):
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
        "brightness":     (R + G + B + NIR) / 4.0,
        "greenness_ratio": G / (R + G + B + 1e-8),
        "canopy_in_shrub_range": ((CHM >= SHRUB_MIN_H) & (CHM <= SHRUB_MAX_H)).astype(np.float32),
        "canopy_shrub_clipped":  np.clip(CHM, 0, SHRUB_MAX_H),
    }

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


def build_pixel_rows(json_path, split_name):
    rng = np.random.default_rng(42)
    with open(json_path) as f:
        dicts = json.load(f)

    feat_names = None
    X_chunks, y_chunks, site_chunks = [], [], []

    print(f"  [{split_name}] {len(dicts)} patches — extracting features...")
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

        label_map = np.full((H, W), -1, dtype=np.int8)
        for cat_id in [1, 2, 0]:
            for ann in anns:
                if ann.get("iscrowd", 0) or ann.get("category_id", 0) != cat_id:
                    continue
                m = poly_to_mask(ann["segmentation"], H, W)
                label_map[m] = 1 if cat_id == 0 else 0

        pos_idx = np.argwhere(label_map == 1)
        neg_idx = np.argwhere(label_map == 0)
        if len(pos_idx) == 0 and len(neg_idx) == 0:
            continue

        if len(neg_idx) > MAX_NEG_PER_PATCH:
            neg_idx = neg_idx[rng.choice(len(neg_idx), MAX_NEG_PER_PATCH, replace=False)]

        idx = np.concatenate([pos_idx, neg_idx], axis=0)
        ys, xs = idx[:, 0], idx[:, 1]
        labels = label_map[ys, xs].astype(np.int8)

        feat_matrix = np.stack([feats[f][ys, xs] for f in feat_names], axis=1).astype(np.float32)
        X_chunks.append(feat_matrix)
        y_chunks.append(labels)
        site_chunks.append(np.full(len(labels), d.get("site", "unknown"), dtype=object))

    X_arr = np.concatenate(X_chunks, axis=0)
    y_arr = np.concatenate(y_chunks, axis=0)
    s_arr = np.concatenate(site_chunks, axis=0)

    print(f"  [{split_name}] Done in {time.time()-t0:.0f}s — "
          f"{len(y_arr):,} pixels  pos={y_arr.sum():,} ({y_arr.mean()*100:.1f}%)")

    return pd.DataFrame(X_arr, columns=feat_names), pd.Series(y_arr, dtype=int), pd.Series(s_arr)


def scale(X_df, features, no_scale_features, scaler):
    scale_cols = [c for c in features if c not in no_scale_features]
    X = X_df[features].copy().astype(np.float32)
    X[scale_cols] = X[scale_cols].fillna(0.0)
    X[scale_cols] = scaler.transform(X[scale_cols].values)
    for c in features:
        if c in no_scale_features:
            X[c] = X[c].fillna(0.0)
    return X


# ---------------------------------------------------------------------------
# Load model + metadata
# ---------------------------------------------------------------------------
print("Loading model and metadata...")
meta    = json.loads(META_PATH.read_text())
model   = joblib.load(MODEL_PATH)
scaler  = joblib.load(SCALER_PATH)

features        = meta["features"]
no_scale_feats  = set(meta.get("no_scale_features", []))
opt_threshold   = meta["cls_threshold"]
saved_metrics   = meta["metrics"]

print(f"  Optuna threshold : {opt_threshold}")
print(f"  Saved AUC        : {saved_metrics['auc']}")
print(f"  Saved F1         : {saved_metrics['f1']}")
print(f"  Top features     : {features}")

# ---------------------------------------------------------------------------
# Build validation set
# ---------------------------------------------------------------------------
print("\nBuilding validation pixel dataset...")
X_val_raw, y_val, sites_val = build_pixel_rows(VAL_JSON, "val")
X_val = scale(X_val_raw, features, no_scale_feats, scaler)

# ---------------------------------------------------------------------------
# Predict probabilities
# ---------------------------------------------------------------------------
print("Running model predictions...")
val_proba = model.predict_proba(X_val)[:, 1]

# ---------------------------------------------------------------------------
# 1. Threshold sweep table
# ---------------------------------------------------------------------------
print("\nComputing threshold sweep...")
rows = []
for thr in THRESHOLD_SWEEP:
    preds = (val_proba >= thr).astype(int)
    p  = precision_score(y_val, preds, zero_division=0)
    r  = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    acc = accuracy_score(y_val, preds)
    iou = jaccard_score(y_val, preds, zero_division=0)
    rows.append({"threshold": thr, "precision": p, "recall": r,
                 "f1": f1, "accuracy": acc, "iou": iou})

# Also add Optuna threshold row if not already in sweep
if opt_threshold not in THRESHOLD_SWEEP:
    preds = (val_proba >= opt_threshold).astype(int)
    rows.append({
        "threshold": opt_threshold,
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall":    recall_score(y_val, preds, zero_division=0),
        "f1":        f1_score(y_val, preds, zero_division=0),
        "accuracy":  accuracy_score(y_val, preds),
        "iou":       jaccard_score(y_val, preds, zero_division=0),
    })

sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
sweep_df.to_csv(OUT_DIR / "threshold_table.csv", index=False)

print("\n" + "="*70)
print("THRESHOLD SWEEP — V12 XGBoost Shrub Classifier")
print("="*70)
print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Accuracy':>9}  {'IoU':>7}")
print("-"*70)
for _, row in sweep_df.iterrows():
    marker = " <-- Optuna" if abs(row.threshold - opt_threshold) < 0.001 else ""
    print(f"{row.threshold:>10.4f}  {row.precision:>10.4f}  {row.recall:>8.4f}  "
          f"{row.f1:>8.4f}  {row.accuracy:>9.4f}  {row.iou:>7.4f}{marker}")
print("="*70)

# ---------------------------------------------------------------------------
# 2. ROC curve
# ---------------------------------------------------------------------------
fpr, tpr, roc_thrs = roc_curve(y_val, val_proba)
auc_val = roc_auc_score(y_val, val_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc_val:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
# Mark Optuna threshold point
opt_preds = (val_proba >= opt_threshold).astype(int)
opt_fpr = (opt_preds[y_val == 0] == 1).mean()
opt_tpr = (opt_preds[y_val == 1] == 1).mean()
ax.scatter([opt_fpr], [opt_tpr], color="#F44336", zorder=5,
           s=80, label=f"Optuna thr={opt_threshold:.4f}")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — V12 XGBoost", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "roc_curve.png", dpi=150)
plt.close(fig)
print(f"Saved roc_curve.png  (AUC={auc_val:.4f})")

# ---------------------------------------------------------------------------
# 3. Precision-Recall curve
# ---------------------------------------------------------------------------
prec_c, rec_c, pr_thrs = precision_recall_curve(y_val, val_proba)
f1_curve = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
best_idx = int(np.argmax(f1_curve))
best_pr_thr = float(pr_thrs[best_idx]) if best_idx < len(pr_thrs) else opt_threshold

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_c, prec_c, color="#4CAF50", lw=2)
ax.scatter([rec_c[best_idx]], [prec_c[best_idx]], color="#FF9800", zorder=5, s=80,
           label=f"Best F1={f1_curve[best_idx]:.4f}  thr={best_pr_thr:.4f}")
ax.scatter(
    [recall_score(y_val, opt_preds, zero_division=0)],
    [precision_score(y_val, opt_preds, zero_division=0)],
    color="#F44336", zorder=5, s=80, marker="D",
    label=f"Optuna thr={opt_threshold:.4f}"
)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — V12 XGBoost", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "pr_curve.png", dpi=150)
plt.close(fig)
print(f"Saved pr_curve.png   (best F1 thr={best_pr_thr:.4f})")

# ---------------------------------------------------------------------------
# 4. Confusion matrix at Optuna threshold
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_val, opt_preds)
tn, fp, fn, tp = cm.ravel()
total = len(y_val)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Non-shrub (pred)", "Shrub (pred)"], fontsize=10)
ax.set_yticklabels(["Non-shrub (true)", "Shrub (true)"], fontsize=10)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix — threshold={opt_threshold:.4f}", fontsize=12, fontweight="bold")

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = val / total * 100
        ax.text(j, i, f"{val:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)
print(f"Saved confusion_matrix.png  (TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,})")

# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({"feature": features, "importance": importances})\
    .sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2196F3"] * len(feat_imp_df)
ax.barh(feat_imp_df["feature"], feat_imp_df["importance"], color=colors, edgecolor="white")
ax.set_xlabel("XGBoost Feature Importance (gain)", fontsize=11)
ax.set_title("Feature Importance — V12 Final Model (Top 20)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
print("Saved feature_importance.png")

# ---------------------------------------------------------------------------
# 6. Threshold sweep plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sweep_df["threshold"], sweep_df["f1"],        lw=2, label="F1",        color="#2196F3")
ax.plot(sweep_df["threshold"], sweep_df["precision"],  lw=2, label="Precision", color="#4CAF50")
ax.plot(sweep_df["threshold"], sweep_df["recall"],     lw=2, label="Recall",    color="#FF9800")
ax.plot(sweep_df["threshold"], sweep_df["iou"],        lw=2, label="IoU",       color="#9C27B0", linestyle="--")
ax.axvline(opt_threshold, color="#F44336", linestyle=":", lw=1.5,
           label=f"Optuna thr={opt_threshold:.4f}")
ax.set_xlabel("Decision Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Metrics vs Threshold — V12 XGBoost Shrub Classifier", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "threshold_sweep.png", dpi=150)
plt.close(fig)
print("Saved threshold_sweep.png")

# ---------------------------------------------------------------------------
# 7. Summary printout
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("  V12 XGBOOST MODEL — EVALUATION SUMMARY")
print("="*55)
print(f"  Val pixels       : {len(y_val):,}")
print(f"  Shrub pixels     : {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
print(f"  AUC-ROC          : {auc_val:.4f}")
print()
print(f"  At Optuna threshold ({opt_threshold:.4f}):")
print(f"    Precision : {precision_score(y_val, opt_preds, zero_division=0):.4f}")
print(f"    Recall    : {recall_score(y_val, opt_preds, zero_division=0):.4f}")
print(f"    F1        : {f1_score(y_val, opt_preds, zero_division=0):.4f}")
print(f"    Accuracy  : {accuracy_score(y_val, opt_preds):.4f}")
print(f"    IoU       : {jaccard_score(y_val, opt_preds, zero_division=0):.4f}")
print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
print()
print(f"  Best PR-curve F1 : {f1_curve[best_idx]:.4f}  (thr={best_pr_thr:.4f})")
print("="*55)
print(f"\nAll outputs saved to: {OUT_DIR}")

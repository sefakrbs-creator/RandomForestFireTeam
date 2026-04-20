"""
viz_ucsd.py
===========
Updated visualization for UCSD AOI with categorical mask on black background and legends.
"""

import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import json

def read_rgb(naip_path):
    """Matches the standard V12 pipeline RGB stretch."""
    with rasterio.open(naip_path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    for c in range(3):
        # 2-98 percentile stretch
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        if hi > lo:
            rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
        else:
            rgb[..., c] = 0
    return rgb

def visualize_results(ucsd_dir):
    dir_path = Path(ucsd_dir)
    naip_path = dir_path / "naip.tif"
    mask_path = dir_path / "predicted_shrub_v12.tif"
    prob_path = dir_path / "predicted_shrub_v12_prob.tif"
    meta_path = Path("v12_model_meta.json")
    
    # Load metadata for threshold
    threshold = 0.9  # Set to 0.9 as requested by user

    print(f"Generating enhanced visualization for {ucsd_dir}...")
    
    # 1. Load Data
    rgb = read_rgb(naip_path)
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        
    with rasterio.open(prob_path) as src:
        prob = src.read(1)
        # Handle nodata (-1.0)
        prob_viz = np.ma.masked_where(prob == -1.0, prob)

    H, W = mask.shape
    shrub_px = int(np.sum(mask == 1))
    shrub_ha = shrub_px / 10000.0
    
    # Probability stats for title
    valid_probs = prob[prob != -1.0]
    p50 = np.percentile(valid_probs, 50) if valid_probs.size > 0 else 0
    p95 = np.percentile(valid_probs, 95) if valid_probs.size > 0 else 0

    # 2. Setup Figure
    aspect = H / W
    fig_w = 15
    img_h = fig_w / 3 * aspect
    fig_h = max(img_h + 1.2, 5.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    
    # Panel 0: RGB
    axes[0].imshow(rgb, aspect="auto")
    axes[0].set_title("UCSD AOI\nNAIP RGB (2-98% stretch)", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 1: Probability (Using a "Black-based" colormap as requested)
    # Using 'magma' or 'viridis' is better for black backgrounds, 
    # but 'RdYlGn' is specifically "yellow and green" which the user disliked.
    # I will use a custom Black -> Yellow -> Green colormap or just 'viridis'.
    # Actually, the user said "background color should be black not yellow or green".
    # I'll use 'magma' for probability as it starts at black.
    im = axes[1].imshow(prob_viz, cmap="magma", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Shrub Probability", fontsize=9)
    axes[1].set_title(f"V12 Confidence Heatmap\np50={p50:.3f}  p95={p95:.3f}", fontsize=10)
    axes[1].axis("off")
    
    # Panel 2: Mask on Black Background
    # Create a categorical map: 0=Black, 1=Green
    mask_viz = np.zeros((H, W, 3), dtype=np.float32)
    shrub_color = [0.0, 1.0, 0.4] # Vibrant Green
    non_shrub_color = [0.0, 0.0, 0.0] # Black
    
    mask_viz[mask == 1] = shrub_color
    mask_viz[mask == 0] = non_shrub_color
    
    axes[2].imshow(mask_viz, aspect="auto")
    axes[2].set_title(
        f"Final Binary Classification\n{shrub_ha:.2f} ha shrub predicted",
        fontsize=10
    )
    axes[2].axis("off")
    
    # Add Legend to Panel 2
    shrub_patch = mpatches.Patch(color=shrub_color, label='Shrub')
    non_shrub_patch = mpatches.Patch(color=non_shrub_color, label='Non-Shrub')
    axes[2].legend(handles=[shrub_patch, non_shrub_patch], 
                   loc='lower right', fontsize=9, facecolor='white', framealpha=0.8)
    
    fig.suptitle(f"V12 XGBoost Shrub Detection -- UCSD Test AOI ({H}x{W} px)",
                 fontsize=12, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    viz_out = dir_path / "ucsd_prediction_viz.png"
    plt.savefig(viz_out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Visualization saved to {viz_out}")

if __name__ == "__main__":
    visualize_results("Model_Prediction/ucsd")

import json

NB = 'C:/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2/shrub_pipeline.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 21 index (the Results Visualisation code cell)
target_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'all_sites_overview' in src and cell['cell_type'] == 'code':
        target_idx = i
        break

print(f"Found viz cell at index: {target_idx}")

lines = [
    'import matplotlib.pyplot as plt\n',
    'import matplotlib.gridspec as gridspec\n',
    '\n',
    'log_section("7. Visualisation -- Site by Site")\n',
    '\n',
    'def read_rgb(naip_path):\n',
    '    with rasterio.open(naip_path) as src:\n',
    '        r = src.read(1).astype(np.float32)\n',
    '        g = src.read(2).astype(np.float32)\n',
    '        b = src.read(3).astype(np.float32)\n',
    '    rgb = np.stack([r, g, b], axis=-1)\n',
    '    for c in range(3):\n',
    '        lo, hi = np.percentile(rgb[..., c], [2, 98])\n',
    '        rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1) if hi > lo else 0\n',
    '    return rgb\n',
    '\n',
    'ok_results = [r for r in results if r["status"] == "OK"]\n',
    '\n',
    '# -- One figure per site --------------------------------------------------\n',
    'site_paths = []\n',
    'for r in ok_results:\n',
    '    site   = r["site"]\n',
    '    naip_p = BASE / site / "NAIP_3DEP_product" / SITES[site]["naip"]\n',
    '    rgb    = read_rgb(naip_p)\n',
    '    prob   = r["prob_map"]\n',
    '    mask   = r["mask"]\n',
    '    H, W   = mask.shape\n',
    '\n',
    '    aspect  = H / W\n',
    '    fig_w   = 15\n',
    '    img_h   = fig_w / 3 * aspect\n',
    '    fig_h   = max(img_h + 1.0, 3.5)\n',
    '\n',
    '    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))\n',
    '\n',
    '    axes[0].imshow(rgb, aspect="auto")\n',
    '    axes[0].set_title(f"{site}\\nNAIP RGB", fontsize=10, fontweight="bold")\n',
    '    axes[0].axis("off")\n',
    '\n',
    '    im = axes[1].imshow(prob, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")\n',
    '    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)\n',
    '    axes[1].set_title(f"V12 Probability\\np50={r[\'p50\']}  p95={r[\'p95\']}", fontsize=10)\n',
    '    axes[1].axis("off")\n',
    '\n',
    '    overlay = np.zeros((H, W, 4), dtype=np.float32)\n',
    '    overlay[mask == 1] = [0.0, 1.0, 0.3, 0.45]\n',
    '    axes[2].imshow(rgb, aspect="auto")\n',
    '    axes[2].imshow(overlay, aspect="auto")\n',
    '    axes[2].set_title(\n',
    '        f"Binary Mask (thr={meta[\'cls_threshold\']})\\n{r[\'shrub_ha\']} ha shrub",\n',
    '        fontsize=10\n',
    '    )\n',
    '    axes[2].axis("off")\n',
    '\n',
    '    fig.suptitle(f"V12 XGBoost Shrub Detection -- {site}  ({H}x{W} px)",\n',
    '                 fontsize=12, fontweight="bold", y=1.01)\n',
    '    plt.tight_layout()\n',
    '\n',
    '    site_png = OUT_ROOT / site / f"{site.lower()}_v12_overview.png"\n',
    '    plt.savefig(site_png, dpi=150, bbox_inches="tight", facecolor="white")\n',
    '    site_paths.append(site_png)\n',
    '    plt.show()\n',
    '    log(f"  [{site}] saved overview: {site_png.name}")\n',
    '\n',
    '# -- Combined overview -----------------------------------------------------\n',
    'n = len(ok_results)\n',
    'fig2, axes2 = plt.subplots(n, 3, figsize=(15, 4 * n))\n',
    'if n == 1:\n',
    '    axes2 = [axes2]\n',
    'for row, r in enumerate(ok_results):\n',
    '    site   = r["site"]\n',
    '    naip_p = BASE / site / "NAIP_3DEP_product" / SITES[site]["naip"]\n',
    '    rgb    = read_rgb(naip_p)\n',
    '    prob   = r["prob_map"]\n',
    '    mask   = r["mask"]\n',
    '    axes2[row][0].imshow(rgb)\n',
    '    axes2[row][0].set_title(f"{site}\\nNAIP RGB", fontsize=8)\n',
    '    axes2[row][0].axis("off")\n',
    '    im = axes2[row][1].imshow(prob, cmap="RdYlGn", vmin=0, vmax=1)\n',
    '    plt.colorbar(im, ax=axes2[row][1], fraction=0.046, pad=0.04)\n',
    '    axes2[row][1].set_title(f"V12 Probability\\np50={r[\'p50\']}  p95={r[\'p95\']}", fontsize=8)\n',
    '    axes2[row][1].axis("off")\n',
    '    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)\n',
    '    overlay[mask == 1] = [0.0, 1.0, 0.3, 0.45]\n',
    '    axes2[row][2].imshow(rgb)\n',
    '    axes2[row][2].imshow(overlay)\n',
    '    axes2[row][2].set_title(\n',
    '        f"Binary Mask (thr={meta[\'cls_threshold\']})\\n{r[\'shrub_ha\']} ha shrub",\n',
    '        fontsize=8\n',
    '    )\n',
    '    axes2[row][2].axis("off")\n',
    'fig2.suptitle("V12 XGBoost Shrub Detection -- All Sites", fontsize=12, y=1.01)\n',
    'fig2.tight_layout()\n',
    'overview_path = OUT_ROOT / "all_sites_overview.png"\n',
    'fig2.savefig(overview_path, dpi=150, bbox_inches="tight")\n',
    'plt.show()\n',
    'log(f"  Saved combined overview : {overview_path}")\n',
    '\n',
    '# -- Probability histograms ------------------------------------------------\n',
    'fig3, axes3 = plt.subplots(2, 3, figsize=(14, 7))\n',
    'axes3 = axes3.flatten()\n',
    'for i, r in enumerate(ok_results):\n',
    '    ax = axes3[i]\n',
    '    ax.hist(r["prob_map"].flatten(), bins=50, color="steelblue", edgecolor="none", alpha=0.8)\n',
    '    ax.axvline(meta["cls_threshold"], color="red", linestyle="--",\n',
    '               label=f"thr={meta[\'cls_threshold\']}")\n',
    '    ax.set_title(r["site"], fontsize=9)\n',
    '    ax.set_xlabel("Shrub probability")\n',
    '    ax.set_ylabel("Pixel count")\n',
    '    ax.legend(fontsize=8)\n',
    'for j in range(len(ok_results), 6):\n',
    '    axes3[j].set_visible(False)\n',
    'fig3.suptitle("V12 Probability Distributions -- All Sites", fontsize=11)\n',
    'fig3.tight_layout()\n',
    'hist_path = OUT_ROOT / "probability_histograms.png"\n',
    'fig3.savefig(hist_path, dpi=150, bbox_inches="tight")\n',
    'plt.show()\n',
    'log(f"  Saved histograms: {hist_path}")\n',
]

nb['cells'][target_idx]['source'] = lines

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f'Cell {target_idx} fixed successfully')

# Verify: join and check for literal newlines inside strings
src = ''.join(lines)
try:
    compile(src, '<cell>', 'exec')
    print('Syntax check: OK')
except SyntaxError as e:
    print(f'Syntax check FAILED: {e}')

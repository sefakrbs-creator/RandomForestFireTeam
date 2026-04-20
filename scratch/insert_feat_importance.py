import json

NB = '/mnt/c/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2/shrub_pipeline.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 17 (training code) — insert AFTER it
train_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if '5. V12 Training' in src and cell['cell_type'] == 'code':
        train_idx = i
        break
print(f'Training cell at index: {train_idx}')

# --- Markdown header cell ---
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 5b. Feature Importance\n",
        "Top-20 features selected by XGBoost Pass-1 importance from `v12_model_meta.json`, "
        "colour-coded by feature group."
    ]
}

# --- Code cell: horizontal bar chart ---
lines = []
lines.append('import json\n')
lines.append('import matplotlib.pyplot as plt\n')
lines.append('import matplotlib.patches as mpatches\n')
lines.append('\n')
lines.append('log_section("5b. Feature Importance")\n')
lines.append('\n')
lines.append('META_PATH = BASE / "v12_model_meta.json"\n')
lines.append('with open(META_PATH) as _f:\n')
lines.append('    _meta = json.load(_f)\n')
lines.append('\n')
lines.append('# Top-20 selected features (in training order) + their Pass-1 importance\n')
lines.append('_selected = set(_meta["features"])\n')
lines.append('_ranking  = [e for e in _meta["feature_selection"]["pass1_importance_ranking"]\n')
lines.append('             if e["feature"] in _selected]\n')
lines.append('_ranking  = sorted(_ranking, key=lambda e: e["importance"])  # ascending for barh\n')
lines.append('\n')
lines.append('_features = [e["feature"] for e in _ranking]\n')
lines.append('_scores   = [e["importance"] for e in _ranking]\n')
lines.append('\n')
lines.append('# Colour scheme by feature group\n')
lines.append('def _feat_color(name):\n')
lines.append('    if name.startswith("canopy_height"):\n')
lines.append('        return "#1565C0"   # dark blue  -- canopy height texture\n')
lines.append('    if name in ("canopy_shrub_clipped", "canopy_in_shrub_range"):\n')
lines.append('        return "#42A5F5"   # light blue -- canopy derived\n')
lines.append('    if name.startswith(("naip_red", "naip_green", "naip_nir", "naip_blue")):\n')
lines.append('        return "#2E7D32"   # dark green -- NAIP bands / texture\n')
lines.append('    return "#E65100"       # orange     -- spectral indices / texture\n')
lines.append('\n')
lines.append('_colors = [_feat_color(f) for f in _features]\n')
lines.append('\n')
lines.append('fig, ax = plt.subplots(figsize=(10, 7))\n')
lines.append('bars = ax.barh(_features, _scores, color=_colors, edgecolor="white", height=0.7)\n')
lines.append('\n')
lines.append('# Value labels\n')
lines.append('for bar, score in zip(bars, _scores):\n')
lines.append('    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,\n')
lines.append('            f"{score:.4f}", va="center", ha="left", fontsize=8)\n')
lines.append('\n')
lines.append('# Legend\n')
lines.append('legend_items = [\n')
lines.append('    mpatches.Patch(color="#1565C0", label="Canopy height texture (mean/std)"),\n')
lines.append('    mpatches.Patch(color="#42A5F5", label="Canopy height derived"),\n')
lines.append('    mpatches.Patch(color="#2E7D32", label="NAIP band / texture"),\n')
lines.append('    mpatches.Patch(color="#E65100", label="Spectral index / texture"),\n')
lines.append(']\n')
lines.append('ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.9)\n')
lines.append('\n')
lines.append('ax.set_xlabel("XGBoost feature importance (Pass-1)", fontsize=10)\n')
lines.append('ax.set_title(\n')
lines.append('    f"V12 Top-{len(_features)} Feature Importance\\n"\n')
lines.append('    f"(selected from {_meta[\'feature_selection\'][\'candidate_count\']} candidates)",\n')
lines.append('    fontsize=12, fontweight="bold"\n')
lines.append(')\n')
lines.append('ax.spines[["top", "right"]].set_visible(False)\n')
lines.append('ax.set_xlim(0, max(_scores) * 1.18)\n')
lines.append('plt.tight_layout()\n')
lines.append('\n')
lines.append('feat_imp_path = OUT_ROOT / "v12_feature_importance.png"\n')
lines.append('plt.savefig(feat_imp_path, dpi=150, bbox_inches="tight", facecolor="white")\n')
lines.append('plt.show()\n')
lines.append('log(f"  Saved feature importance chart: {feat_imp_path}")\n')
lines.append('\n')
lines.append('# Summary stats\n')
lines.append('_canopy_imp = sum(s for f, s in zip(_features, _scores)\n')
lines.append('                  if f.startswith("canopy_height") or\n')
lines.append('                  f in ("canopy_shrub_clipped", "canopy_in_shrub_range"))\n')
lines.append('log(f"  Canopy-height group cumulative importance: {_canopy_imp:.1%}")\n')
lines.append('log(f"  Top feature: {_features[-1]}  ({_scores[-1]:.4f})")\n')

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": lines
}

# Insert markdown + code cell after train_idx
nb['cells'].insert(train_idx + 1, md_cell)
nb['cells'].insert(train_idx + 2, code_cell)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f'Inserted feature importance cells at positions {train_idx+1} and {train_idx+2}')
print(f'Total cells: {len(nb["cells"])}')

# Syntax check
src = ''.join(lines)
try:
    compile(src, '<cell>', 'exec')
    print('Syntax check: OK')
except SyntaxError as e:
    print(f'Syntax check FAILED: {e}')

import json

NB = 'C:/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2/shrub_pipeline.ipynb'
with open(NB) as f:
    nb = json.load(f)

# Box-drawing chars
TL,TR,BL,BR = '\u250c','\u2510','\u2514','\u2518'
H,V,LT,RT,TT,BT,CR = '\u2500','\u2502','\u251c','\u2524','\u252c','\u2534','\u253c'
W1, W2 = 21, 7

lines = [
    'import subprocess, sys\n',
    'from pathlib import Path\n',
    '\n',
    'log_section("Generate All SAM Tile PNGs")\n',
    '\n',
    'SCRIPT  = BASE / "generate_all_sam_viz.py"\n',
    'OUT_DIR = BASE / "sam_viz_all"\n',
    '\n',
    '# \u2500\u2500 Tile count table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
    'SITE_TILES = [\n',
    '    ("Calaveras_Big_trees", 463),\n',
    '    ("DL_Bliss",             40),\n',
    '    ("Independence_Lake",   132),\n',
    '    ("Pacific_Union",        53),\n',
    '    ("Sedgwick",            151),\n',
    '    ("Shaver_Lake",          40),\n',
    ']\n',
    'total_tiles = sum(n for _, n in SITE_TILES)\n',
    '\n',
    f'log("SAM tile counts per site (train + val combined):")\n',
    f'log(f"  {TL}{H*W1}{TT}{H*W2}{TR}")\n',
    f'log(f"  {V} {{\'Site\':<{W1-2}}} {V} {{\'Tiles\':>{W2-2}}} {V}")\n',
    f'log(f"  {LT}{H*W1}{CR}{H*W2}{RT}")\n',
    'for i, (site, n) in enumerate(SITE_TILES):\n',
    f'    log(f"  {V} {{site:<{W1-2}}} {V} {{n:>{W2-2}}} {V}")\n',
    '    if i < len(SITE_TILES) - 1:\n',
    f'        log(f"  {LT}{H*W1}{CR}{H*W2}{RT}")\n',
    f'log(f"  {LT}{H*W1}{CR}{H*W2}{RT}")\n',
    f'log(f"  {V} {{\'Total\':<{W1-2}}} {V} {{total_tiles:>{W2-2}}} {V}")\n',
    f'log(f"  {BL}{H*W1}{BT}{H*W2}{BR}")\n',
    'log("")\n',
    '\n',
    '# \u2500\u2500 Run script if outputs not yet present \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
    'if not SCRIPT.exists():\n',
    '    log(f"  Script not found: {SCRIPT}", level="warning")\n',
    'elif not OUT_DIR.exists():\n',
    '    log("  Running generate_all_sam_viz.py (both splits, all sites)...")\n',
    '    import time as _time\n',
    '    t0 = _time.time()\n',
    '    result = subprocess.run(\n',
    '        [sys.executable, str(SCRIPT),\n',
    '         "--out-dir", str(OUT_DIR),\n',
    '         "--splits", "train", "val",\n',
    '         "--dpi", "100", "--grid-cols", "6"],\n',
    '        capture_output=True, text=True\n',
    '    )\n',
    '    elapsed = _time.time() - t0\n',
    '    print(result.stdout)\n',
    '    if result.returncode != 0:\n',
    '        print(result.stderr)\n',
    '        log(f"  ERROR (exit {result.returncode})", level="error")\n',
    '    else:\n',
    '        log(f"  Done in {elapsed:.0f}s")\n',
    'else:\n',
    '    log(f"  sam_viz_all/ already exists ({OUT_DIR.resolve()}) \u2014 skipping re-generation")\n',
    '\n',
    '# \u2500\u2500 Show summary grid \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
    'summary_grid = OUT_DIR / "summary_grid.png"\n',
    'if summary_grid.exists():\n',
    '    import matplotlib.pyplot as plt\n',
    '    import matplotlib.image as mpimg\n',
    '    fig, ax = plt.subplots(figsize=(14, 8))\n',
    '    ax.imshow(mpimg.imread(summary_grid))\n',
    '    ax.axis("off")\n',
    '    ax.set_title(\n',
    '        f"SAM Viz \u2014 summary_grid.png  (4 patches per site \u00d7 {len(SITE_TILES)} sites)\\n"\n',
    '        f"Total tiles in dataset: {total_tiles}  |  green=shrub  red=tree  blue=rock",\n',
    '        fontsize=10, fontweight="bold"\n',
    '    )\n',
    '    plt.tight_layout()\n',
    '    plt.show()\n',
    '    log(f"  Outputs: {OUT_DIR.resolve()}")\n',
    'else:\n',
    '    log(f"  summary_grid.png not found at {OUT_DIR} \u2014 run generate_all_sam_viz.py first",\n',
    '        level="warning")\n',
]

nb['cells'][14]['source'] = lines

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print('Cell 14 updated successfully')
print(f'Total cells: {len(nb["cells"])}')

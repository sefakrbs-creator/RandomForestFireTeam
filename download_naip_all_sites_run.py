"""
Download fresh 1m NAIP (2022) from Planetary Computer for all 6 sites.
Outputs saved to <Site>/NAIP_3DEP_product/<site>_1m_naip_2022.tif
"""

import sys
sys.path.insert(0, '/mnt/c/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2')

from download_naip_all_sites import download_naip_aoi

BASE = '/mnt/c/Users/sefak/OneDrive/Documents/sefakarabas/OneDrive/Desktop/Dataset-2'

SITES = [
    {
        'name': 'Calaveras_Big_trees',
        'bbox': [-120.263288, 38.235991, -120.212972, 38.258505],
        'output': f'{BASE}/Calaveras_Big_trees/NAIP_3DEP_product/calaveras_big_trees_1m_naip_2022.tif',
    },
    {
        'name': 'DL_Bliss',
        'bbox': [-120.105333, 38.989207, -120.095954, 38.999506],
        'output': f'{BASE}/DL_Bliss/NAIP_3DEP_product/dl_bliss_1m_naip_2022.tif',
    },
    {
        'name': 'Independence_Lake',
        'bbox': [-120.348706, 39.410336, -120.256253, 39.476121],
        'output': f'{BASE}/Independence_Lake/NAIP_3DEP_product/independence_lake_1m_naip_2022.tif',
    },
    {
        'name': 'Pacific_Union',
        'bbox': [-122.424811, 38.563895, -122.404202, 38.581865],
        'output': f'{BASE}/Pacific_Union/NAIP_3DEP_product/pacific_union_college_1m_naip_2022.tif',
    },
    {
        'name': 'Sedgwick',
        'bbox': [-120.059646, 34.676532, -120.038295, 34.697095],
        'output': f'{BASE}/Sedgwick/NAIP_3DEP_product/sedgwick_1m_naip_2022.tif',
    },
    {
        'name': 'Shaver_Lake',
        'bbox': [-119.285897, 37.094483, -119.265648, 37.110254],
        'output': f'{BASE}/Shaver_Lake/NAIP_3DEP_product/shaver_lake_1m_naip_2022.tif',
    },
]

for site in SITES:
    print(f"\n{'='*60}")
    print(f"Downloading NAIP for: {site['name']}")
    print(f"{'='*60}")
    try:
        download_naip_aoi(
            bbox_wgs84=site['bbox'],
            mask_crs='EPSG:6350',
            target_res=1.0,
            year='2022',
            output_path=site['output'],
        )
        print(f"  SUCCESS: {site['output']}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nAll done.")

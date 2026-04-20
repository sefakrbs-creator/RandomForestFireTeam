import rasterio
from pathlib import Path

sites = {
    "Calaveras_Big_trees": "calaveras_big_trees_1m_naip_2022.tif",
    "DL_Bliss": "dl_bliss_1m_naip_2022.tif",
    "Independence_Lake": "independence_lake_1m_naip_2022.tif",
    "Pacific_Union": "pacific_union_college_1m_naip_2022.tif",
    "Sedgwick": "sedgwick_1m_naip_2022.tif",
    "Shaver_Lake": "shaver_lake_1m_naip_2022.tif",
}

for site, naip_file in sites.items():
    p = Path(site) / "NAIP_3DEP_product" / naip_file
    if p.exists():
        with rasterio.open(p) as src:
            print(f"{site}: {src.width}x{src.height} ({src.count} bands)")
    else:
        print(f"{site}: File not found at {p}")

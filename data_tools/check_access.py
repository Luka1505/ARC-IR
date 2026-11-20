"""Run this FIRST on the GPU machine: can it reach Planetary Computer and read Landsat + WorldCover pixels?
    python -m data_tools.check_access"""
import sys
ok = True
def step(name, fn):
    global ok
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:
        ok = False; print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
try:
    from pystac_client import Client; import planetary_computer as pc, rasterio
except ImportError as e:
    sys.exit(f"missing package: {e}.  pip install pystac-client planetary-computer rasterio stackstac pyproj")
cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
bbox = [80.6, 15.7, 81.1, 16.2]; holder = {}
def s1():
    it = next(cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime="2023-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": 20}}, max_items=1).items())
    holder["it"] = it; return f"{it.id}, cloud {it.properties['eo:cloud_cover']}%, sun elevation {it.properties['view:sun_elevation']}"
def s2():
    with rasterio.open(holder["it"].assets["lwir11"].href) as src:
        w = src.read(1, window=((2000, 2064), (2000, 2064))); return f"read a 64x64 thermal window, dtype {w.dtype}, max {w.max()}"
def s3():
    it = next(cat.search(collections=["esa-worldcover"], bbox=bbox, datetime="2021-01-01/2021-12-31", max_items=1).items())
    holder["wc"] = it; return f"{it.id}, assets {list(it.assets)}"
def s4():
    with rasterio.open(holder["wc"].assets["map"].href) as src:
        w = src.read(1, window=((1000, 1064), (1000, 1064))); return f"read a 64x64 WorldCover window, classes {sorted(set(w.flatten().tolist()))}"
step("STAC search Landsat", s1)
if "it" in holder: step("read thermal pixels", s2)
step("STAC search WorldCover", s3)
if "wc" in holder: step("read WorldCover pixels", s4)
print("\nALL GOOD: you can start the download." if ok else "\nSomething failed: send me the FAIL lines.")

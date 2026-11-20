"""Find which read method works on this PC:  python -m data_tools.debug_read"""
import time, traceback, numpy as np, rasterio, planetary_computer as pc
from pystac_client import Client
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from data_tools.make_tiles import STAC, get_epsg, bbox_to_bounds
cat = Client.open(STAC, modifier=pc.sign_inplace)
bbox = [80.6, 15.7, 81.1, 16.2]
it = next(cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime="2023-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": 20}}, max_items=1).items())
href = it.assets["lwir11"].href; epsg = get_epsg(it); b = bbox_to_bounds(bbox, epsg)
print("scene", it.id, "epsg", epsg, "bounds", b)
def t(name, fn):
    s = time.time()
    try: r = fn(); print(f"[ OK ] {name}: {r}  ({time.time()-s:.0f}s)")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}"); c = e.__cause__ or e.__context__
        if c: print("        cause:", type(c).__name__, c)
def a():
    with rasterio.open(href) as s: return f"crs {s.crs}, size {s.width}x{s.height}, transform {tuple(s.transform)[:6]}"
def w():
    with rasterio.open(href) as s: return s.read(1, window=((1000, 1500), (1000, 1500))).shape
def v():
    with rasterio.open(href) as s:
        with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(b[0], b[3], 30, 30), width=300, height=300, src_nodata=0, nodata=0) as vrt:
            return vrt.read(1).shape
t("open", a); t("plain window read 500x500", w); t("WarpedVRT read 300x300", v)

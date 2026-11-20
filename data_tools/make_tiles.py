"""Download Landsat 8/9 Collection 2 Level-2 scenes + ESA WorldCover from Microsoft Planetary Computer and cut tiles
into  dataset/<region>/*.npz  (the folder `dataset` next to this code).

RESUMABLE: a scene that finished is skipped on the next run, a scene that fails is logged and skipped, so you can
stop (Ctrl-C) and restart any time. Run it in the background:
    nohup python -m data_tools.make_tiles > dataset/download.log 2>&1 &
    tail -f dataset/download.log
Status at any time:   python -m data_tools.make_tiles --summary

STATUS OF THIS CODE: the pure functions (tiling, colour conversion, scene selection, bbox conversion) are unit-tested offline
(python -m common.test_data_tools). The network functions (fetch_scene, fetch_worldcover) were NOT run against the live
service from the build sandbox: run ONE scene first:   python -m data_tools.make_tiles --only krishna_delta --max-scenes 1
Needs: pip install pystac-client planetary-computer rasterio pyproj
"""
import argparse, glob, json, os, shutil, time, traceback, numpy as np, yaml

# WorldCover codes -> our classes: 0 water, 1 forest/shrub, 2 cropland/grass, 3 bare/sparse, 4 built-up
WC_MAP = {10: 1, 20: 1, 30: 2, 40: 2, 50: 4, 60: 3, 70: 3, 80: 0, 90: 0, 95: 0, 100: 3}
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ----------------------------------------------------------------------------- pure functions (tested offline)
def map_worldcover(wc):
    out = np.full(wc.shape, 255, np.uint8)
    for k, v in WC_MAP.items():
        out[wc == k] = v
    return out


def rgb_to_lab(rgb):
    """rgb (3,H,W) display-referred in [0,1] -> L/100 in [0,1], ab/128 roughly in [-1,1] (D65)."""
    c = np.moveaxis(rgb, 0, -1).astype(np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L, a, b = 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])
    return (L / 100).astype(np.float32), (np.stack([a, b]) / 128).astype(np.float32)


def lab_to_rgb(L, ab):
    """inverse of rgb_to_lab, for previews. L (H,W) in [0,1], ab (2,H,W) in about [-1,1] -> rgb (H,W,3) in [0,1]"""
    Lc, a, b = L * 100, ab[0] * 128, ab[1] * 128
    fy = (Lc + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    finv = lambda t: np.where(t ** 3 > 216 / 24389, t ** 3, (116 * t - 16) / (24389 / 27))
    xyz = np.stack([finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883], -1)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    lin = np.clip(xyz @ np.linalg.inv(M).T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).clip(0, 1)


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi]); return np.clip((x - a) / (b - a + 1e-8), 0, 1)


def pick_spread(items, n):
    """n scenes spread evenly over time (items sorted by date), so a place gets several different dates"""
    items = sorted(items, key=lambda it: it.properties["datetime"])
    if len(items) <= n:
        return items
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
    return [items[i] for i in idx]


def get_epsg(item):
    """EPSG code of a scene. Newer STAC uses 'proj:code' ("EPSG:32644"), older uses 'proj:epsg' (32644)."""
    p = item.properties
    if p.get("proj:epsg") is not None:
        return int(p["proj:epsg"])
    code = p.get("proj:code")
    if code and str(code).upper().startswith("EPSG:"):
        return int(str(code).split(":")[1])
    for a in item.assets.values():
        e = a.extra_fields.get("proj:epsg")
        if e is not None:
            return int(e)
        c = a.extra_fields.get("proj:code")
        if c and str(c).upper().startswith("EPSG:"):
            return int(str(c).split(":")[1])
    raise KeyError(f"no projection code on {item.id}: properties={list(p)}")


def bbox_to_bounds(bbox, epsg):
    """lon/lat box -> (xmin, ymin, xmax, ymax) in the projected CRS (every scene of a region uses the SAME grid)"""
    from pyproj import Transformer
    t = Transformer.from_crs(4326, int(epsg), always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])])
    return (min(xs) // 30 * 30, min(ys) // 30 * 30, max(xs) // 30 * 30, max(ys) // 30 * 30)     # snap to the 30 m grid


FIXED_GAIN = 0.30      # reflectance that maps to 1.0 in the fixed colour target (same for every scene and channel)


def fixed_color(rgb_refl, valid):
    """colour target WITHOUT per-scene stretching: reflectance / 0.30, gamma 2.2.  Keeps real colour ratios and brightness, so forest is green and
    soil is brown in every scene (the old target stretched each channel on its own, which pushed every scene towards grey).  -> L (H,W), ab (2,H,W)"""
    x = np.clip(np.where(valid[None], rgb_refl, np.nan) / FIXED_GAIN, 0, 1) ** (1 / 2.2)
    return rgb_to_lab(np.nan_to_num(x).astype(np.float32))


def cut_tiles(thermal_k, rgb_refl, wc_cls, valid, meta, region, scene, out_dir, size=128, min_valid=0.95):
    """thermal_k (H,W) Kelvin | rgb_refl (3,H,W) reflectance | wc_cls (H,W) uint8 | valid (H,W) bool | meta (4,)"""
    os.makedirs(out_dir, exist_ok=True)
    th = np.nan_to_num(stretch(np.where(valid, thermal_k, np.nan))).astype(np.float32)
    tc = np.where(valid, thermal_k - 273.15, 0.0).astype(np.float16)      # ABSOLUTE surface temperature (deg C) before the per-scene stretch
    rgb = np.stack([stretch(np.where(valid, rgb_refl[i], np.nan)) for i in range(3)]) ** (1 / 2.2)
    L, ab = rgb_to_lab(np.nan_to_num(rgb).astype(np.float32))
    Lf, abf = fixed_color(rgb_refl, valid)
    H, W = th.shape; n = 0
    for y in range(0, H - size + 1, size):
        for x in range(0, W - size + 1, size):
            sl = (slice(y, y + size), slice(x, x + size)); lab = wc_cls[sl]
            if valid[sl].mean() < min_valid or (lab == 255).mean() > 0.05:
                continue
            final = os.path.join(out_dir, f"{scene}_y{y}_x{x}.npz"); tmp = os.path.join(out_dir, f"_tmp_{scene}_y{y}_x{x}.npz")
            np.savez_compressed(tmp, thermal=th[sl], thermal_c=tc[sl], L_fixed=Lf[sl].astype(np.float16), ab_fixed=abf[:, sl[0], sl[1]].astype(np.float16), L=L[sl], ab=ab[:, sl[0], sl[1]],
                                lab=np.where(lab == 255, 0, lab).astype(np.uint8), meta=np.asarray(meta, np.float32),
                                region=region, scene=scene, tile_xy=f"{region}_y{y}_x{x}")
            os.replace(tmp, final)                                   # a tile appears under its final name only when complete
            n += 1
    return n


# ----------------------------------------------------------------------------- network functions (UNTESTED live)
def retry(fn, tries=3, wait=20):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            cause = e.__cause__ or e.__context__
            print(f"      attempt {k + 1}/{tries} failed: {type(e).__name__}: {str(e)[:300]}" + (f" | cause: {type(cause).__name__}: {str(cause)[:300]}" if cause else ""), flush=True)
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def warp_read(href, epsg, bounds, resolution=30, resampling="nearest", nodata=0, dtype=np.float64):
    """read one raster onto the region's fixed grid (rasterio only reads the needed window of the cloud file)"""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    xmin, ymin, xmax, ymax = bounds
    W, H = int(round((xmax - xmin) / resolution)), int(round((ymax - ymin) / resolution))
    # HTTP/1.1 + no multiplexing + many retries: some networks/proxies cut partial (206) downloads short ("got N bytes, expected M")
    env = dict(GDAL_HTTP_TIMEOUT="90", GDAL_HTTP_MAX_RETRY="10", GDAL_HTTP_RETRY_DELAY="3", GDAL_HTTP_VERSION="1.1",
               GDAL_HTTP_MULTIPLEX="NO", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="NO", CPL_VSIL_CURL_CACHE_SIZE="200000000",
               CPL_VSIL_CURL_CHUNK_SIZE="1048576")   # fewer, bigger requests: the server is far away, so each request costs waiting time
    last = None
    for k in range(4):                       # a truncated read is retried on its own, so one bad block does not redo all 5 bands
        try:
            with rasterio.Env(**env):
                with rasterio.open(href) as src:
                    with WarpedVRT(src, crs=f"EPSG:{int(epsg)}", transform=from_origin(xmin, ymax, resolution, resolution), width=W, height=H,
                                   resampling=getattr(Resampling, resampling), src_nodata=nodata, nodata=nodata) as vrt:
                        return vrt.read(1).astype(dtype)
        except Exception as e:
            last = e; time.sleep(5 * (k + 1))
    raise last


def fresh_item(cat, item):
    """look the scene up again so its signed download link is new (links expire after about an hour, a long run then fails with HTTP 403)"""
    try:
        found = list(cat.search(collections=["landsat-c2-l2"], ids=[item.id]).items())
        return found[0] if found else item
    except Exception:
        return item


def fetch_scene(item, bounds, epsg):
    """load red/green/blue/lwir11/qa_pixel at 30 m on the region's fixed grid"""
    import planetary_computer as pc
    from concurrent.futures import ThreadPoolExecutor
    t = time.time(); print("      reading Landsat bands (5 files)...", flush=True)
    item = pc.sign(item)
    names = ["red", "green", "blue", "lwir11", "qa_pixel"]
    with ThreadPoolExecutor(1) as ex:
        arrs = list(ex.map(lambda b: warp_read(item.assets[b].href, epsg, bounds, 30, "nearest", 0), names))
    print(f"      Landsat read done in {time.time() - t:.0f}s, grid {arrs[0].shape}", flush=True)
    red, green, blue, lw, qa = [np.where(a == 0, np.nan, a) if n != "qa_pixel" else a for a, n in zip(arrs, names)]
    refl = np.stack([red, green, blue]) * 0.0000275 - 0.2             # Collection 2 L2 surface reflectance scaling
    temp = lw * 0.00341802 + 149.0                                     # Collection 2 L2 surface temperature (K)
    qa = np.nan_to_num(qa).astype(np.int64)
    bad = ((qa >> 0) & 1) | ((qa >> 1) & 1) | ((qa >> 2) & 1) | ((qa >> 3) & 1) | ((qa >> 4) & 1)  # fill, dilated cloud, cirrus, cloud, shadow
    valid = (bad == 0) & np.isfinite(temp) & np.isfinite(refl).all(0) & (temp > 200)
    return refl, temp, valid


def fetch_worldcover(bbox, bounds, epsg, shape):
    """ESA WorldCover 2021 (collection 'esa-worldcover', asset 'map') on the same grid, nearest-neighbour"""
    import planetary_computer as pc
    from pystac_client import Client
    t = time.time(); print("      reading WorldCover...", flush=True)
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=bbox, datetime="2021-01-01/2021-12-31").items())
    if not items:
        raise RuntimeError("no WorldCover 2021 items for " + str(bbox))
    wc = np.zeros(shape, np.float64)
    for i in items:
        a = warp_read(pc.sign(i).assets["map"].href, epsg, bounds, 30, "nearest", 0)
        wc = np.where(wc == 0, a, wc)
    print(f"      WorldCover done in {time.time() - t:.0f}s", flush=True)
    if wc.shape != shape:
        raise RuntimeError(f"WorldCover grid {wc.shape} != scene grid {shape}")
    return wc


def meta_vector(item):
    import datetime as dt
    d = dt.datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00")); doy = d.timetuple().tm_yday / 365.25
    return [np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy), item.properties["view:sun_elevation"] / 90.0, 0.0]   # terrain height not used yet


# ----------------------------------------------------------------------------- driver
def summary(out):
    tot, byreg = 0, {}
    for r in sorted(glob.glob(os.path.join(out, "*", ""))):
        n = len(glob.glob(os.path.join(r, "[!_]*.npz"))); sc = len(glob.glob(os.path.join(r, "_done_*")))
        byreg[os.path.basename(os.path.dirname(r))] = (n, sc); tot += n
    gb = sum(os.path.getsize(f) for f in glob.glob(os.path.join(out, "*", "[!_]*.npz"))) / 1e9
    for k, (n, sc) in byreg.items():
        print(f"  {k:28s} {n:6d} tiles from {sc:3d} scenes")
    print(f"TOTAL {tot} tiles, {gb:.2f} GB, {len(byreg)} regions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="data_tools/regions.yaml"); ap.add_argument("--out", default="dataset")
    ap.add_argument("--only", default=None); ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--part", default=None, help='split regions over several windows, e.g. "1/4" ... "4/4"')
    ap.add_argument("--summary", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summary:
        return summary(a.out)
    free = shutil.disk_usage(a.out).free / 1e9
    print(f"free disk: {free:.0f} GB" + ("  (WARNING: under 30 GB)" if free < 30 else ""), flush=True)
    cfg = yaml.safe_load(open(a.regions)); from pystac_client import Client; import planetary_computer as pc
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    part = {r: p for p in ("train", "calibration", "test") for r in cfg[p]}
    t0 = time.time()
    for region, bbox in cfg["bbox"].items():
        if a.part:
            i, n = map(int, a.part.split("/"))
            if list(cfg["bbox"]).index(region) % n != i - 1:
                continue
        if a.only and region != a.only:
            continue
        rdir = os.path.join(a.out, region)
        search = cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=cfg.get("datetime", "2021-01-01/2025-12-31"),
                            query={"eo:cloud_cover": {"lt": cfg["max_cloud_percent"]}, "platform": {"in": ["landsat-8", "landsat-9"]}})
        items = pick_spread(list(search.items()), a.max_scenes or cfg["max_scenes_per_region"])
        print(f"[{region} | {part.get(region, '?')}] {len(items)} scenes", flush=True)
        if not items:
            continue
        epsg = get_epsg(items[0]); bounds = bbox_to_bounds(bbox, epsg); wc = None
        os.makedirs(rdir, exist_ok=True)
        json.dump({"epsg": int(epsg), "bounds": [float(b) for b in bounds]}, open(os.path.join(rdir, "_grid.json"), "w"))   # lets data_tools.enrich reuse the exact grid
        shape = (int(round((bounds[3] - bounds[1]) / 30)), int(round((bounds[2] - bounds[0]) / 30)))
        wc_file = os.path.join(rdir, "_worldcover.npy")          # land cover is fetched ONCE per region and kept on disk
        try:
            if os.path.exists(wc_file):
                wc = np.load(wc_file)
            else:
                wc = map_worldcover(retry(lambda: fetch_worldcover(bbox, bounds, epsg, shape), tries=5, wait=15)); np.save(wc_file, wc)
        except Exception as e:
            print(f"   WorldCover FAILED for {region}: {type(e).__name__}: {str(e)[:200]}  (region skipped, rerun to retry)", flush=True); continue
        for it in items:
            mark = os.path.join(rdir, f"_done_{it.id}")
            if os.path.exists(mark):
                print(f"   {it.id}: already done", flush=True); continue
            raw = os.path.join(rdir, f"_raw_{it.id}.npz")        # the slow Landsat read is kept, so a later failure never repeats it
            try:
                if os.path.exists(raw):
                    z = np.load(raw); refl, temp, valid = z["refl"], z["temp"], z["valid"]
                else:
                    refl, temp, valid = retry(lambda: fetch_scene(fresh_item(cat, it), bounds, epsg)); np.savez(raw, refl=refl, temp=temp, valid=valid)
                n = cut_tiles(temp, refl, wc, valid, meta_vector(it), region, it.id, rdir, cfg["tile_size"], cfg.get("min_valid_fraction", 0.95))
                open(mark, "w").write(str(n))
                try: os.remove(raw)
                except OSError: pass
                print(f"   {it.id}: {n} tiles  ({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
            except Exception as e:
                print(f"   {it.id}: FAILED {type(e).__name__}: {str(e)[:200]}  (skipped, rerun to retry)", flush=True)
                traceback.print_exc()
    print("finished"); summary(a.out)


if __name__ == "__main__":
    main()

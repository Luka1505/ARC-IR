"""Adds three things to the tiles you already downloaded, WITHOUT downloading any Landsat again:
  1. better land-cover labels: majority class of the 3x3 ten-metre WorldCover pixels inside each 30 m pixel (the old labels sampled ONE of nine)
  2. soft labels: the share (percent) of each class inside each 30 m pixel  -> key 'soft' (K,H,W) uint8
  3. terrain: Copernicus DEM elevation (m) and slope (degrees) at 30 m   -> keys 'dem', 'slope'
Each tile file is rewritten in place (old 'lab' is kept as 'lab_old'). Safe to stop and re-run: tiles that already have 'soft' are skipped.
Run again after the download finishes to enrich the tiles that were added since.
    python -m data_tools.enrich                       # all regions
    python -m data_tools.enrich --only krishna_delta  # one region first
    python -m data_tools.enrich --check krishna_delta # look at one region's arrays without changing anything
"""
import argparse, glob, json, os, re, time, numpy as np, yaml
from data_tools.make_tiles import (STAC, WC_MAP, get_epsg, bbox_to_bounds, warp_read, retry, pick_spread)

K = 5


def soft_labels(wc10, k=K):
    """wc10: (3H,3W) uint8 ESA classes at 10 m on the 30 m grid.  Returns (mode (H,W) uint8, soft (K,H,W) uint8 percent)."""
    mapped = np.full(wc10.shape, 255, np.uint8)
    for c, v in WC_MAP.items():
        mapped[wc10 == c] = v
    H, W = mapped.shape[0] // 3, mapped.shape[1] // 3
    blk = mapped[:H * 3, :W * 3].reshape(H, 3, W, 3)
    cnt = np.stack([(blk == c).sum((1, 3)) for c in range(k)]).astype(np.float32)          # (K,H,W) 0..9
    tot = cnt.sum(0)
    soft = np.round(100 * cnt / np.maximum(tot, 1)).astype(np.uint8)
    mode = np.where(tot > 0, cnt.argmax(0), 0).astype(np.uint8)
    return mode, soft


def slope_deg(dem, res=30.0):
    gy, gx = np.gradient(dem.astype(np.float32), res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def region_grid(rdir, region, bbox, cfg, cat):
    f = os.path.join(rdir, "_grid.json")
    if os.path.exists(f):
        g = json.load(open(f)); return g["epsg"], tuple(g["bounds"])
    search = cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=cfg.get("datetime", "2021-01-01/2025-12-31"),
                        query={"eo:cloud_cover": {"lt": cfg["max_cloud_percent"]}, "platform": {"in": ["landsat-8", "landsat-9"]}})
    items = pick_spread(list(search.items()), cfg["max_scenes_per_region"])
    epsg = get_epsg(items[0]); bounds = bbox_to_bounds(bbox, epsg)
    json.dump({"epsg": int(epsg), "bounds": [float(b) for b in bounds]}, open(f, "w"))
    return epsg, bounds


def fetch_enrich(bbox, bounds, epsg):
    import planetary_computer as pc
    from pystac_client import Client
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=bbox, datetime="2021-01-01/2021-12-31").items())
    H, W = int(round((bounds[3] - bounds[1]) / 30)), int(round((bounds[2] - bounds[0]) / 30))
    wc = np.zeros((H * 3, W * 3), np.uint8)
    for i in items:
        a = warp_read(pc.sign(i).assets["map"].href, epsg, bounds, 10, "nearest", 0, dtype=np.uint8)
        wc = np.where(wc == 0, a, wc)
    mode, soft = soft_labels(wc)
    dem = np.full((H, W), np.nan, np.float32)
    for i in cat.search(collections=["cop-dem-glo-30"], bbox=bbox).items():
        d = warp_read(pc.sign(i).assets["data"].href, epsg, bounds, 30, "bilinear", -32768, dtype=np.float32)
        dem = np.where(np.isnan(dem) & (d != -32768), d, dem)
    dem = np.nan_to_num(dem, nan=0.0)
    return mode, soft, dem.astype(np.float16), slope_deg(dem).astype(np.float16)


def patch_tiles(rdir, mode, soft, dem, slope, log=print):
    n = skipped = 0
    for f in sorted(glob.glob(os.path.join(rdir, "[!_]*.npz"))):
        m = re.search(r"_y(\d+)_x(\d+)\.npz$", f)
        if not m:
            continue
        y, x = int(m.group(1)), int(m.group(2))
        try:
            with np.load(f, allow_pickle=False) as z:   # close the handle: Windows cannot replace an open file
                if "soft" in z.files:
                    skipped += 1; continue
                d = {k: z[k] for k in z.files}
        except Exception as e:
            log(f"    skipping unreadable {os.path.basename(f)}: {e}"); continue
        h, w = d["thermal"].shape; sl = (slice(y, y + h), slice(x, x + w))
        d["lab_old"] = d["lab"]; d["lab"] = mode[sl]; d["soft"] = soft[:, sl[0], sl[1]]; d["dem"] = dem[sl]; d["slope"] = slope[sl]
        tmp = os.path.join(rdir, "_tmp_" + os.path.basename(f))
        np.savez_compressed(tmp, **d)
        for attempt in range(10):          # antivirus / indexer can briefly lock a new file
            try:
                os.replace(tmp, f); break
            except PermissionError:
                if attempt == 9: raise
                time.sleep(0.5)
        n += 1
    return n, skipped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--regions", default="data_tools/regions.yaml"); ap.add_argument("--out", default="dataset")
    ap.add_argument("--only", default=None); ap.add_argument("--check", default=None); a = ap.parse_args()
    import planetary_computer as pc
    from pystac_client import Client
    cfg = yaml.safe_load(open(a.regions)); cat = Client.open(STAC, modifier=pc.sign_inplace); t0 = time.time()
    for region, bbox in cfg["bbox"].items():
        if (a.only and region != a.only) or (a.check and region != a.check):
            continue
        rdir = os.path.join(a.out, region)
        if not glob.glob(os.path.join(rdir, "[!_]*.npz")):
            continue
        ef = os.path.join(rdir, "_enrich.npz")
        try:
            if os.path.exists(ef):
                z = np.load(ef); mode, soft, dem, slope = z["mode"], z["soft"], z["dem"], z["slope"]
            else:
                epsg, bounds = region_grid(rdir, region, bbox, cfg, cat)
                mode, soft, dem, slope = retry(lambda: fetch_enrich(bbox, bounds, epsg), tries=5, wait=15)
                old = os.path.join(rdir, "_worldcover.npy")
                if os.path.exists(old) and np.load(old).shape != mode.shape:
                    print(f"[{region}] WARNING grid {mode.shape} differs from the download grid {np.load(old).shape}: skipped", flush=True); continue
                np.savez_compressed(ef, mode=mode, soft=soft, dem=dem, slope=slope)
            if a.check:
                print(f"[{region}] mode {mode.shape}, class shares {np.bincount(mode.ravel(), minlength=K) / mode.size}, "
                      f"dem {float(dem.min()):.0f}..{float(dem.max()):.0f} m, slope max {float(slope.max()):.1f} deg"); continue
            n, sk = patch_tiles(rdir, mode, soft, dem, slope)
            print(f"[{region}] patched {n} tiles, {sk} already done  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        except Exception as e:
            print(f"[{region}] FAILED {type(e).__name__}: {str(e)[:200]}  (rerun to retry)", flush=True)
    print("finished")


if __name__ == "__main__":
    main()
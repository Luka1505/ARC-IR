"""Adds the FIXED-SCALE colour target (keys 'L_fixed', 'ab_fixed') and, where missing, the absolute temperature ('thermal_c') to tiles that were cut earlier.
Reads red, green, blue and QA of each scene (plus lwir11 only for tiles that still lack 'thermal_c'), then patches the tiles in place.
    python -m data_tools.backfill_color --regions data_tools/regions.yaml --out dataset                   # all regions
    python -m data_tools.supervise --module data_tools.backfill_color --parts 6 --regions data_tools/regions.yaml --out dataset
Safe to stop and re-run: a scene is skipped once its '_color_<scene>' marker exists.  Use the result with  --color-target fixed  (and --abs-temp)."""
import argparse, glob, json, os, re, time, numpy as np, yaml
from data_tools.make_tiles import STAC, warp_read, retry, fixed_color


def read_bands(cat, scene, epsg, bounds, need_lwir):
    import planetary_computer as pc
    items = list(cat.search(collections=["landsat-c2-l2"], ids=[scene]).items())
    if not items:
        raise RuntimeError(f"scene {scene} not found in the catalogue")
    it = pc.sign(items[0]); names = ["red", "green", "blue", "qa_pixel"] + (["lwir11"] if need_lwir else [])
    arr = [warp_read(it.assets[b].href, epsg, bounds, 30, "nearest", 0) for b in names]
    return np.stack(arr[:3]), arr[3], (arr[4] if need_lwir else None)


def scene_color(raw, qa):
    """fixed-scale colour of the whole scene, using the same cloud / fill mask as make_tiles"""
    refl = np.where(raw == 0, np.nan, raw * 0.0000275 - 0.2)
    q = np.nan_to_num(qa).astype(np.int64); bad = ((q >> 0) & 1) | ((q >> 1) & 1) | ((q >> 2) & 1) | ((q >> 3) & 1) | ((q >> 4) & 1)
    return fixed_color(refl, (bad == 0) & np.isfinite(refl).all(0))


def temperature_c(lw):
    return np.where(lw > 0, lw * 0.00341802 + 149.0 - 273.15, 0.0).astype(np.float16)


def tiles_of(rdir, scene):
    for f in sorted(glob.glob(os.path.join(rdir, f"{scene}_y*_x*.npz"))):
        m = re.search(r"_y(\d+)_x(\d+)\.npz$", f)
        if m and os.path.basename(f)[:-len(m.group(0))] == scene:
            yield f, int(m.group(1)), int(m.group(2))


def needs_lwir(rdir, scene):
    for f, _, _ in tiles_of(rdir, scene):
        with np.load(f, allow_pickle=False) as z:
            if "thermal_c" not in z.files:
                return True
    return False


def patch_scene(rdir, scene, Lf, abf, temp_c=None, log=print):
    """-> (patched, already_done, min correlation between the new and the old brightness of a tile (alignment check))"""
    n = had = 0; minc = 1.0
    for f, y, x in tiles_of(rdir, scene):
        try:
            with np.load(f, allow_pickle=False) as z:          # close the handle: Windows cannot replace an open file
                d = {k: z[k] for k in z.files}
        except Exception as e:
            log(f"    skipping unreadable {os.path.basename(f)}: {e}"); continue
        if "ab_fixed" in d and "thermal_c" in d:
            had += 1; continue
        h, w = d["thermal"].shape; sl = (slice(y, y + h), slice(x, x + w))
        if Lf[sl].shape != (h, w):
            raise RuntimeError(f"scene grid {Lf.shape} does not cover tile {os.path.basename(f)}")
        if "thermal_c" not in d:
            if temp_c is None:
                raise RuntimeError("a tile lacks thermal_c but no thermal band was read")
            d["thermal_c"] = temp_c[sl]
        ok = d["thermal_c"] != 0                               # pixels with no valid temperature were set to 0 in the tiles (and are 0 in the colour too)
        Lt, abt = Lf[sl] * ok, abf[:, sl[0], sl[1]] * ok
        d["L_fixed"], d["ab_fixed"] = Lt.astype(np.float16), abt.astype(np.float16)
        v = ok & (d["L"] > 0)
        if v.sum() > 100 and Lt[v].std() > 0 and d["L"][v].std() > 0:
            minc = min(minc, float(np.corrcoef(Lt[v].ravel(), d["L"][v].ravel())[0, 1]))
        tmp = os.path.join(rdir, "_tmp_" + os.path.basename(f)); np.savez_compressed(tmp, **d)
        for attempt in range(10):                              # antivirus / indexer can briefly lock a new file
            try:
                os.replace(tmp, f); break
            except PermissionError:
                if attempt == 9: raise
                time.sleep(0.5)
        n += 1
    return n, had, minc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--regions", default="data_tools/regions.yaml"); ap.add_argument("--out", default="dataset")
    ap.add_argument("--only", default=None); ap.add_argument("--part", default=None); a = ap.parse_args()
    import planetary_computer as pc
    from pystac_client import Client
    cfg = yaml.safe_load(open(a.regions)); cat = Client.open(STAC, modifier=pc.sign_inplace); t0 = time.time()
    for ri, region in enumerate(cfg["bbox"]):
        if a.part:
            i, n = map(int, a.part.split("/"))
            if ri % n != i - 1:
                continue
        if a.only and region != a.only:
            continue
        rdir = os.path.join(a.out, region)
        if not os.path.exists(os.path.join(rdir, "_grid.json")):
            continue
        g = json.load(open(os.path.join(rdir, "_grid.json"))); epsg, bounds = g["epsg"], tuple(g["bounds"])
        scenes = sorted({re.sub(r"_y\d+_x\d+\.npz$", "", os.path.basename(f)) for f in glob.glob(os.path.join(rdir, "[!_]*.npz"))})
        print(f"[{region}] {len(scenes)} scenes", flush=True)
        for sc in scenes:
            mark = os.path.join(rdir, f"_color_{sc}")
            if os.path.exists(mark):
                print(f"   {sc}: already done", flush=True); continue
            try:
                lw_needed = needs_lwir(rdir, sc)
                raw, qa, lw = retry(lambda: read_bands(cat, sc, epsg, bounds, lw_needed), tries=4, wait=15)
                Lf, abf = scene_color(raw.astype(np.float64), qa)
                n, had, minc = patch_scene(rdir, sc, Lf, abf, temperature_c(lw) if lw is not None else None)
                open(mark, "w").write(str(n))
                warn = f"  WARNING: new and old brightness correlate only {minc:.2f} (misaligned?)" if minc < 0.8 else ""
                print(f"   {sc}: patched {n} tiles ({had} already had it)  ({(time.time() - t0) / 60:.0f} min elapsed){warn}", flush=True)
            except Exception as e:
                print(f"   {sc}: FAILED {type(e).__name__}: {str(e)[:150]}  (skipped, rerun to retry)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

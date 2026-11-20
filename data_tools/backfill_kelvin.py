"""Adds ABSOLUTE surface temperature to tiles that were cut before it was stored  -> key 'thermal_c' (degrees C, float16), the input of --abs-temp.
It downloads ONE band per scene (lwir11), not five, so it is several times faster than a new download; tiles are patched in place.
    python -m data_tools.backfill_kelvin --regions data_tools/regions_big.yaml --out dataset_big            # all regions
    python -m data_tools.backfill_kelvin ... --part 1/4                                                      # one of 4 windows
    python -m data_tools.supervise --module data_tools.backfill_kelvin --parts 4 --regions data_tools/regions_big.yaml --out dataset_big   # 4 windows + restarts
Safe to stop and re-run: a scene is skipped once all its tiles have 'thermal_c' (and a '_kelvin_<scene>' marker exists)."""
import argparse, glob, json, os, re, time, numpy as np, yaml
from data_tools.make_tiles import STAC, warp_read, retry


def patch_scene(rdir, scene, temp_c, log=print):
    """write temp_c[y:y+h, x:x+w] into every tile of this scene that lacks 'thermal_c'; returns (patched, already_had_it)"""
    n = had = 0
    for f in sorted(glob.glob(os.path.join(rdir, f"{scene}_y*_x*.npz"))):
        m = re.search(r"_y(\d+)_x(\d+)\.npz$", f)
        if not m or os.path.basename(f)[:-len(m.group(0))] != scene:
            continue
        y, x = int(m.group(1)), int(m.group(2))
        try:
            with np.load(f, allow_pickle=False) as z:          # close the handle: Windows cannot replace an open file
                if "thermal_c" in z.files:
                    had += 1; continue
                d = {k: z[k] for k in z.files}
        except Exception as e:
            log(f"    skipping unreadable {os.path.basename(f)}: {e}"); continue
        h, w = d["thermal"].shape; piece = temp_c[y:y + h, x:x + w]
        if piece.shape != (h, w):
            raise RuntimeError(f"temperature grid {temp_c.shape} does not cover tile {os.path.basename(f)}")
        d["thermal_c"] = piece.astype(np.float16)
        tmp = os.path.join(rdir, "_tmp_" + os.path.basename(f)); np.savez_compressed(tmp, **d)
        for attempt in range(10):                              # antivirus / indexer can briefly lock a new file
            try:
                os.replace(tmp, f); break
            except PermissionError:
                if attempt == 9: raise
                time.sleep(0.5)
        n += 1
    return n, had


def read_temperature(cat, scene, epsg, bounds):
    """the scene's lwir11 band on the region grid -> degrees C (0 where there is no data).  The scene is looked up again so its signed link is fresh."""
    import planetary_computer as pc
    items = list(cat.search(collections=["landsat-c2-l2"], ids=[scene]).items())
    if not items:
        raise RuntimeError(f"scene {scene} not found in the catalogue")
    lw = warp_read(pc.sign(items[0]).assets["lwir11"].href, epsg, bounds, 30, "nearest", 0)
    return np.where(lw > 0, lw * 0.00341802 + 149.0 - 273.15, 0.0).astype(np.float16)


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
            mark = os.path.join(rdir, f"_kelvin_{sc}")
            if os.path.exists(mark):
                print(f"   {sc}: already done", flush=True); continue
            try:
                with_k = [f for f in glob.glob(os.path.join(rdir, f"{sc}_y*_x*.npz"))]
                temp_c = retry(lambda: read_temperature(cat, sc, epsg, bounds), tries=4, wait=15)
                n, had = patch_scene(rdir, sc, temp_c)
                open(mark, "w").write(str(n))
                print(f"   {sc}: patched {n} tiles ({had} already had it)  ({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
            except Exception as e:
                print(f"   {sc}: FAILED {type(e).__name__}: {str(e)[:150]}  (skipped, rerun to retry)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

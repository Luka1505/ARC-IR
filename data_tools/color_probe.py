"""Is the COLOUR TARGET the reason colour accuracy does not improve?   (small download: about 12 scenes, 4 bands each)

The tiles were cut with every colour channel stretched on its own to 0..1 per scene (2nd..98th percentile).  That forces each channel to the same average,
so a scene's colour cast, and the real differences between classes (green forest, brown bare soil, dark blue water), are partly erased: your palette check showed
all five class colours within +-0.04 of grey, so even a perfect classifier cannot predict colour.
This probe downloads the red, green, blue and QA bands of a few scenes and compares three colour targets on the SAME tiles and labels:
   old    every channel stretched on its own  (what the tiles have now)
   fixed  one fixed gain for every scene and channel (reflectance / 0.30), gamma 2.2   -> keeps real colour ratios and real brightness
   joint  one stretch shared by the three channels of a scene (exposure fixed, hue kept)
and prints how much the class colour explains of each ("class gain" = 1 - error with the true class colour / error with one global colour).
    python -m data_tools.color_probe --regions data_tools/regions.yaml --out dataset
"""
import argparse, glob, json, os, re, numpy as np, yaml
from concurrent.futures import ThreadPoolExecutor
from data_tools.make_tiles import STAC, warp_read, retry, rgb_to_lab, stretch

BANDS = ["red", "green", "blue", "qa_pixel"]
GAIN = 0.30


def read_raw(cat, scene, epsg, bounds, cache):
    f = os.path.join(cache, scene + ".npz")
    if os.path.exists(f):
        z = np.load(f); return z["raw"], z["qa"]
    import planetary_computer as pc
    items = list(cat.search(collections=["landsat-c2-l2"], ids=[scene]).items())
    if not items:
        raise RuntimeError(f"scene {scene} not found")
    it = pc.sign(items[0]); arr = [warp_read(it.assets[b].href, epsg, bounds, 30, "nearest", 0) for b in BANDS]
    raw, qa = np.stack(arr[:3]).astype(np.uint16), arr[3].astype(np.uint16)
    os.makedirs(cache, exist_ok=True); np.savez_compressed(f, raw=raw, qa=qa)
    return raw, qa


def targets(raw, qa):
    """-> dict version -> (L, ab) for the whole scene, using the same valid mask as make_tiles"""
    refl = np.where(raw == 0, np.nan, raw.astype(np.float64) * 0.0000275 - 0.2)
    q = qa.astype(np.int64); bad = ((q >> 0) & 1) | ((q >> 1) & 1) | ((q >> 2) & 1) | ((q >> 3) & 1) | ((q >> 4) & 1)
    valid = (bad == 0) & np.isfinite(refl).all(0); refl = np.where(valid[None], refl, np.nan)
    old = np.stack([stretch(refl[i]) for i in range(3)])
    fixed = np.clip(refl / GAIN, 0, 1)
    lo, hi = np.nanpercentile(refl, [2, 98]); joint = np.clip((refl - lo) / (hi - lo + 1e-8), 0, 1)
    return {k: rgb_to_lab(np.nan_to_num(v ** (1 / 2.2)).astype(np.float32)) for k, v in [("old", old), ("fixed", fixed), ("joint", joint)]}


def collect(rdir, scene, tg, size=128):
    """class labels + ab of every existing tile of this scene, per colour version"""
    labs, abs_ = [], {k: [] for k in tg}; check = 0.0
    for f in sorted(glob.glob(os.path.join(rdir, f"{scene}_y*_x*.npz"))):
        m = re.search(r"_y(\d+)_x(\d+)\.npz$", f)
        if not m or os.path.basename(f)[:-len(m.group(0))] != scene:
            continue
        y, x = int(m.group(1)), int(m.group(2))
        with np.load(f, allow_pickle=False) as z:
            lab, ab_old = z["lab"], z["ab"]
        labs.append(lab.ravel())
        for k, (L, ab) in tg.items():
            abs_[k].append(ab[:, y:y + size, x:x + size].reshape(2, -1))
        check = max(check, float(np.abs(tg["old"][1][:, y:y + size, x:x + size] - ab_old).max()))
    return labs, abs_, check


def summarize(labs, abs_, names, log=print):
    lab = np.concatenate(labs); res = {}
    log(f"{len(lab):,} pixels. 'class gain' = how much of the colour error one colour per class removes (0 = nothing, 1 = everything)")
    log(f"{'colour target':8s} {'colour spread':>14s} {'error, class colour':>20s} {'class gain':>11s}   class colours (a,b)")
    for k, parts in abs_.items():
        ab = np.concatenate(parts, 1); g = ab.mean(1, keepdims=True); mae_g = float(np.abs(ab - g).mean())
        pal = np.stack([ab[:, lab == c].mean(1) if (lab == c).any() else g[:, 0] for c in range(len(names))])
        mae_c = float(np.abs(ab - pal[lab].T).mean()); gain = 1 - mae_c / mae_g; res[k] = {"spread": mae_g, "class_error": mae_c, "class_gain": gain}
        log(f"{k:8s} {mae_g:14.4f} {mae_c:20.4f} {gain:11.3f}   " + ", ".join(f"{names[c]} {pal[c][0]:+.2f},{pal[c][1]:+.2f}" for c in range(len(names))))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--regions", default="data_tools/regions.yaml"); ap.add_argument("--out", default="dataset")
    ap.add_argument("--n-regions", type=int, default=6); ap.add_argument("--scenes", type=int, default=2); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); names = ["water", "forest", "cropland", "bare", "built"]
    import planetary_computer as pc
    from pystac_client import Client
    cfg = yaml.safe_load(open(a.regions)); cat = Client.open(STAC, modifier=pc.sign_inplace); cache = os.path.join(a.out, "_probe_cache")
    regs = [r for r in cfg["bbox"] if os.path.exists(os.path.join(a.out, r, "_grid.json"))]
    step = max(1, len(regs) // a.n_regions); regs = regs[::step][:a.n_regions]; jobs = []
    for r in regs:
        rdir = os.path.join(a.out, r); g = json.load(open(os.path.join(rdir, "_grid.json")))
        sc = sorted({re.sub(r"_y\d+_x\d+\.npz$", "", os.path.basename(f)) for f in glob.glob(os.path.join(rdir, "[!_]*.npz"))})
        sc = sc[::max(1, len(sc) // a.scenes)][:a.scenes]; jobs += [(r, rdir, s, g["epsg"], tuple(g["bounds"])) for s in sc]
    print(f"{len(jobs)} scenes in {len(regs)} regions: {regs}", flush=True)

    def work(j):
        r, rdir, s, epsg, bounds = j
        raw, qa = retry(lambda: read_raw(cat, s, epsg, bounds, cache), tries=4, wait=15)
        print(f"   {s}: read", flush=True); return collect(rdir, s, targets(raw, qa))
    labs, abs_, chk = [], {"old": [], "fixed": [], "joint": []}, 0.0
    with ThreadPoolExecutor(a.workers) as ex:
        for l, b, c in ex.map(work, jobs):
            labs += l; chk = max(chk, c)
            for k in abs_: abs_[k] += b[k]
    print(f"sanity: my 'old' colour vs the colour stored in the tiles differs by at most {chk:.4f} (should be about 0.00)")
    res = summarize(labs, abs_, names); json.dump(res, open(os.path.join(a.out, "color_probe.json"), "w"), indent=1)
    print("reading: if 'fixed' or 'joint' show a clearly higher class gain than 'old', the colour target should be rebuilt (needs the red/green/blue bands again).")


if __name__ == "__main__":
    main()

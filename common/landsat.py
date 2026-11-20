"""Real-data loader. Reads the .npz tiles written by data_tools/make_tiles.py  (folder: dataset/<region>/*.npz).

Each tile file holds (all on the 30 m grid, normalised):
  thermal (H,W) in [0,1] | L (H,W) lightness/100 | ab (2,H,W) Lab chroma/128 | lab (H,W) uint8 class
  meta (4,) = [sin doy, cos doy, sun_elevation/90, terrain_elevation_norm] | region str | scene str | tile_xy str
Regions listed in regions.yaml under train / calibration / test decide the split (geographic, never random).
Memory: arrays are pre-allocated once (no list + stack copy), about 0.3 MB per 128x128 tile, so 40,000 tiles ~ 12 GB RAM.
"""
import glob, os, random, numpy as np, torch, torch.nn.functional as F, yaml
from .synthetic import degrade

DEFAULT_NAMES = ["water", "forest", "cropland", "bare", "built"]


def _load(files, scale, log=print, dates=1, terrain=False, soft=False, K=5, abs_temp=False, color="stretch"):
    n = len(files)
    Lk, abk = ("L_fixed", "ab_fixed") if color == "fixed" else ("L", "ab")
    z0 = np.load(files[0]); H, W = z0["thermal"].shape
    has_soft, has_terr = "soft" in z0.files, "dem" in z0.files
    if soft and not has_soft:
        raise RuntimeError("--soft needs enriched tiles: run  python -m data_tools.enrich  first")
    if terrain and not has_terr:
        raise RuntimeError("--terrain needs enriched tiles: run  python -m data_tools.enrich  first")
    sf = np.empty((n, K, H, W), np.uint8) if soft else None
    dm = np.empty((n, 2, H, W), np.float16) if terrain else None
    tcb = np.empty((n, 1, H, W), np.float16) if abs_temp else None
    regs, scenes = [], []
    hr = np.empty((n, 1, H, W), np.float32); L = np.empty((n, 1, H, W), np.float32); ab = np.empty((n, 2, H, W), np.float32)
    m = np.empty((n, 4), np.float32); lab = np.empty((n, H, W), np.uint8); ids = []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        hr[i, 0], L[i, 0], ab[i], m[i], lab[i] = z["thermal"], z[Lk], z[abk], z["meta"], z["lab"]; ids.append(str(z["tile_xy"]))
        regs.append(str(z["region"])); scenes.append(str(z["scene"]))
        if soft: sf[i] = z["soft"]
        if terrain: dm[i, 0], dm[i, 1] = z["dem"], z["slope"]
        if abs_temp: tcb[i, 0] = z["thermal_c"]
        if (i + 1) % 5000 == 0: log(f"    loaded {i + 1}/{n} tiles")
    D = {"hr": torch.from_numpy(hr), "L": torch.from_numpy(L), "ab": torch.from_numpy(ab), "m": torch.from_numpy(m), "lab": torch.from_numpy(lab)}
    D["lr"] = torch.cat([degrade(D["hr"][i:i + 2048], scale) for i in range(0, n, 2048)])
    pair = -torch.ones(n, dtype=torch.long); by = {}
    for i, t in enumerate(ids):
        by.setdefault(t, []).append(i)
    for idx in by.values():                      # same place on another date: pair them (needs >= 2 scenes)
        for a, b in zip(idx, idx[1:] + idx[:1]):
            if a != b: pair[a] = b
    D["pair"] = pair
    names = sorted(set(regs)); D["reg"] = torch.tensor([names.index(r) for r in regs], dtype=torch.long); D["reg_names"] = names
    if soft:
        D["soft"] = torch.from_numpy(sf)
    aux = _make_aux(D, by, scenes, ids, dm, scale, dates, terrain, tcb)
    if aux is not None:
        D["aux"] = aux
    return D


def _make_aux(D, by, scenes, ids, dm, scale, dates, terrain, tcb=None):
    """Extra inputs at LR resolution (float16): terrain (elevation, slope) and the thermal image + season of K-1 OTHER dates of the same place.
    Channels: [dem, slope] + for each other date [thermal, sin(doy), cos(doy), valid].  Missing dates are zero with valid = 0."""
    n = D["hr"].shape[0]; ch = []
    if terrain:
        t = torch.from_numpy(dm.astype(np.float32))
        t[:, 0] = (t[:, 0] / 3000.0).clamp(-0.05, 1.5); t[:, 1] = (t[:, 1] / 45.0).clamp(0, 1)
        ch.append(F.avg_pool2d(t, scale) if scale > 1 else t)
    if tcb is not None:                                                       # absolute temperature (the per-scene stretch removed it from 'thermal')
        t = (torch.from_numpy(tcb.astype(np.float32)) - 25.0) / 20.0
        ch.append(F.avg_pool2d(t, scale) if scale > 1 else t)
    if dates > 1:
        lr, m = D["lr"], D["m"]; h, w = lr.shape[-2:]
        others = -torch.ones(n, dates - 1, dtype=torch.long)
        for idx in by.values():
            idx = sorted(idx, key=lambda i: scenes[i])
            for p, i in enumerate(idx):
                for j in range(1, dates):
                    if j < len(idx): others[i, j - 1] = idx[(p + j) % len(idx)]
        for j in range(dates - 1):
            o = others[:, j]; ok = o >= 0; oo = o.clamp(min=0)
            th = lr[oo] * ok[:, None, None, None]
            sc = m[oo][:, :2] * ok[:, None]                                   # sin/cos of the other date's day of year
            ch += [th, sc[:, :, None, None].expand(-1, -1, h, w), ok[:, None, None, None].float().expand(-1, 1, h, w)]
    return torch.cat(ch, 1).half() if ch else None


def build(tiles_dir, regions_yaml, scale=2, num_classes=5, names=None, max_tiles=None, log=print, dates=1, terrain=False, soft=False, abs_temp=False, color="stretch"):
    """max_tiles: optional cap per split (random subset, fixed seed) if RAM is short. Note: capping breaks date pairs for tiles
    whose partner is dropped, so prefer a smaller regions list over a cap."""
    cfg = yaml.safe_load(open(regions_yaml)); out = []; reg_names = {}
    for part in ("train", "calibration", "test"):
        files = []
        for r in cfg[part]:
            files += sorted(glob.glob(os.path.join(tiles_dir, r, "[!_]*.npz")))
        if not files:
            raise FileNotFoundError(f"no tiles for {part} regions {cfg[part]} under {tiles_dir}")
        need = (["thermal_c"] if abs_temp else []) + (["ab_fixed", "L_fixed"] if color == "fixed" else [])
        if need:                                      # older tiles were cut before these arrays were stored: leave them out
            keep = []
            for f in files:
                with np.load(f, allow_pickle=False) as z:
                    if all(k in z.files for k in need): keep.append(f)
            log(f"  {part}: {len(keep)} of {len(files)} tiles have {' + '.join(need)} (--abs-temp / --color-target fixed use only those)")
            files = keep
            if not files:
                raise FileNotFoundError(f"no tile with {need} in {part} regions; run data_tools.backfill_color first")
        if max_tiles and len(files) > max_tiles:
            random.Random(0).shuffle(files); files = sorted(files[:max_tiles])
        log(f"  {part}: {len(files)} tiles")
        D = _load(files, scale, log, dates, terrain, soft, num_classes, abs_temp, color); reg_names[part] = D.pop("reg_names"); out.append(D)
    aux_ch = (2 if terrain else 0) + (1 if abs_temp else 0) + 4 * (dates - 1)
    info = {"K": num_classes, "names": names or DEFAULT_NAMES[:num_classes], "mdim": 4, "aux_ch": aux_ch, "soft": soft, "reg_names": reg_names}
    return out[0], out[1], out[2], info

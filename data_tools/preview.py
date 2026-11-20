"""Look at tiles: thermal | reconstructed RGB | land-cover map, side by side.  python -m data_tools.preview --region krishna_delta [--n 12]"""
import argparse, glob, os, random, numpy as np
from PIL import Image
from .make_tiles import lab_to_rgb
COL = np.array([[40, 90, 200], [30, 130, 50], [200, 190, 80], [190, 150, 110], [200, 60, 60]], np.uint8)   # water forest crop bare built


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="dataset"); ap.add_argument("--region", required=True)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default=None); a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.dir, a.region, "[!_]*.npz"))); random.Random(0).shuffle(fs)
    if not fs: raise SystemExit(f"no tiles in {a.dir}/{a.region}")
    rows = []
    for f in fs[: a.n]:
        z = np.load(f); t = (np.stack([z["thermal"]] * 3, -1) * 255).astype(np.uint8)
        rgb = (lab_to_rgb(z["L"], z["ab"]) * 255).astype(np.uint8); lab = COL[np.clip(z["lab"], 0, 4)]
        rows.append(np.concatenate([t, rgb, lab], 1))
    out = a.out or os.path.join(a.dir, f"preview_{a.region}.png"); Image.fromarray(np.concatenate(rows, 0)).save(out)
    print(f"saved {out}  ({min(a.n, len(fs))} tiles of {len(fs)}; columns: thermal | colour | land cover)")


if __name__ == "__main__":
    main()

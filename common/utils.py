import argparse, json, math, os, time, numpy as np, torch, torch.nn.functional as F
from . import synthetic, landsat
from .models import Seg


def add_common_args(ap):
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "landsat"])
    ap.add_argument("--tiles-dir", default="dataset", help="landsat: folder with <region>/*.npz")
    ap.add_argument("--regions", default="data_tools/regions.yaml")
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--max-tiles", type=int, default=None, help="landsat: cap tiles per split if RAM is short")
    ap.add_argument("--scale", type=int, default=2, help="SR factor (thermal LR -> HR)")
    ap.add_argument("--n", type=int, default=64, help="synthetic tile size")
    ap.add_argument("--tiles", type=int, default=120); ap.add_argument("--ncal", type=int, default=80); ap.add_argument("--ntest", type=int, default=70)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--structure", default="small", choices=["small", "hat", "hat-s"])
    ap.add_argument("--weights", default=None, help="pretrained backbone weights (hat)")
    ap.add_argument("--dates", type=int, default=1, help="landsat: thermal images of K dates of the same place as input (1 = single date)")
    ap.add_argument("--terrain", action="store_true", help="landsat: add elevation + slope channels (needs data_tools.enrich)")
    ap.add_argument("--abs-temp", action="store_true", help="landsat: add the ABSOLUTE surface temperature as an input channel (tiles cut with the new make_tiles only)")
    ap.add_argument("--color-target", default="stretch", choices=["stretch", "fixed"], help="landsat: colour to predict. fixed = one fixed reflectance scale for every scene (tiles with L_fixed/ab_fixed only)")
    ap.add_argument("--soft", action="store_true", help="landsat: train the material head on soft land-cover fractions (needs data_tools.enrich)")
    return ap


def setup(args, name):
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out or os.path.join("runs", name); os.makedirs(out, exist_ok=True)
    lf = open(os.path.join(out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); lf.write(s + "\n"); lf.flush()
    log(f"== {name} | device {dev} | data {args.data} | seed {args.seed}")
    return dev, out, log


def get_data(args, log):
    t0 = time.time()
    if args.data == "synthetic":
        tr, ca, te, info = synthetic.build(args.tiles, args.ncal, args.ntest, args.n, args.scale, args.seed)
        info.setdefault("aux_ch", 0)
    else:
        tr, ca, te, info = landsat.build(args.tiles_dir, args.regions, args.scale, args.classes, max_tiles=args.max_tiles, log=log,
                                       dates=args.dates, terrain=args.terrain, soft=args.soft, abs_temp=args.abs_temp, color=args.color_target)
    log(f"data: train {len(tr['hr'])} cal {len(ca['hr'])} test {len(te['hr'])} tiles, HR {tuple(tr['hr'].shape[1:])} "
        f"LR {tuple(tr['lr'].shape[1:])}, K={info['K']} ({time.time() - t0:.0f}s)")
    return tr, ca, te, info


def to_dev(b, dev):
    return {k: (v.long() if k == "lab" else v).to(dev) for k, v in b.items()}


def batch(D, idx):
    return {k: v[idx] for k, v in D.items()}


def cosine_lr(opt, base, step, steps):
    for g in opt.param_groups:
        g["lr"] = base * 0.5 * (1 + math.cos(math.pi * step / steps))


class Timer:
    """prints seconds/step and an estimate for the full run: use it to extrapolate GPU time from a short test run"""
    def __init__(s, steps, log):
        s.t0, s.steps, s.log = time.time(), steps, log

    def tick(s, step):
        if step in (9, 49) or step == s.steps - 1:
            el = time.time() - s.t0; sps = el / (step + 1)
            s.log(f"    [timer] {step + 1} steps in {el:.1f}s = {sps:.3f} s/step -> {s.steps} steps ~ {sps * s.steps / 60:.1f} min")


def train_seg(D, info, dev, steps=300, seed=123):
    torch.manual_seed(seed)
    seg = Seg(info["K"]).to(dev); opt = torch.optim.AdamW(seg.parameters(), lr=3e-3)
    N = D["L"].shape[0]
    for _ in range(steps):
        i = torch.randint(0, N, (32,)); b = to_dev(batch(D, i), dev)
        loss = F.cross_entropy(seg(b["L"], b["ab"]), b["lab"])
        opt.zero_grad(); loss.backward(); opt.step()
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg.eval()


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=1)

"""Qualitative figures for the paper. Picks test tiles from different regions (random, fixed seed - not hand-picked), runs the saved models on those tiles only and draws
  fig_colour : thermal input | true colour | regression | pix2pix | ARC-IR colour | ARC-IR spread | ARC-IR abs. error
  fig_sr     : bicubic LR | SR baseline | ARC-IR SR | true thermal  (+ zoomed crop)
Needs runs/final_arc_s<seed>, final_reg_s0, final_pix2pix_s0, final_sr_s0 (model.pt). Output: <out>/fig_colour.{png,pdf}, fig_sr.{png,pdf}
Run:  python -m v6_final.make_figures --data landsat --tiles-dir dataset_big --regions data_tools/regions_big.yaml --soft --terrain --dates 3 --abs-temp --color-target fixed --arc-seed 0 --out runs/figures
"""
import argparse, os, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common.utils import add_common_args, setup, get_data
from common.models import ARC
from common.tta import predict_arc, predict_unet
from v1_baselines.run_fair import UNet, make_input


def lab2rgb(L, ab):                                     # L = lightness/100, ab = chroma/128 (the dataset's storage)
    L, a, b = L * 100.0, ab[0] * 128.0, ab[1] * 128.0
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = lambda t: np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    X, Y, Z = 0.95047 * f(fx), 1.0 * f(fy), 1.08883 * f(fz)
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z; g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z; bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    rgb = np.stack([r, g, bl], -1); rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)


def load_arc(run, info, dev):
    ck = torch.load(f"{run}/model.pt", map_location=dev); a = ck["args"]
    m = ARC("arc", info, a["structure"], a["scale"], a["weights"], aux_ch=info["aux_ch"], width_mult=a["width_mult"], hybrid=not a["no_hybrid"]).to(dev); m.load_state_dict(ck["state"]); return m.eval()


def load_unet(run, D, hr, dev):
    ck = torch.load(f"{run}/model.pt", map_location=dev); a = ck["args"]
    cin = make_input({k: D[k][:2] for k in ["lr", "m"] + (["aux"] if "aux" in D else [])}, hr, a["inputs"]).shape[1]
    n = UNet(cin, {"sr": 1, "regvar": 4}.get(a["method"], 2), a["width"]).to(dev); n.load_state_dict(ck["state"]); return n.eval(), a


def main():
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--runs-dir", default="runs"); ap.add_argument("--arc-seed", type=int, default=0); ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args(); dev, out, log = setup(a, "figures"); tr, ca, te, info = get_data(a, log); del tr, ca
    reg = te["reg"].numpy() if "reg" in te else np.zeros(len(te["hr"]), int); rng = np.random.default_rng(0); regions = np.unique(reg); rng.shuffle(regions)
    idx = [int(rng.choice(np.where(reg == r)[0])) for r in regions[:a.rows]]; log(f"tiles {idx} from regions {[int(reg[i]) for i in idx]}")
    S = {k: v[idx] for k, v in te.items() if hasattr(v, "shape") and len(v) == len(te["hr"])}; hr = S["hr"].shape[-2:]
    R = lambda n: f"{a.runs_dir}/final_{n}"
    P = predict_arc(load_arc(R(f"arc_s{a.arc_seed}"), info, dev), S, dev, tta=False, S=16)
    nets = {n: load_unet(R(f"{n}_s0"), S, hr, dev) for n in ("reg", "pix2pix", "sr")}
    U = {n: predict_unet(net, S, dev, make_input, args["method"], args["inputs"], False) for n, (net, args) in nets.items()}
    n = len(idx); os.makedirs(out, exist_ok=True)
    cols = ["Thermal input", "True colour", "Regression", "Pix2pix", "ARC-IR colour", "ARC-IR spread", "ARC-IR |error|"]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.0 * len(cols), 2.0 * n), squeeze=False)
    for i in range(n):
        L = S["L"][i, 0].numpy(); up = F.interpolate(S["lr"][i:i + 1], size=hr, mode="bicubic", align_corners=False)[0, 0].numpy()
        err = (P["ab"][i] - S["ab"][i]).abs().mean(0).numpy()
        panels = [(up, "inferno", None), (lab2rgb(L, S["ab"][i].numpy()), None, None), (lab2rgb(L, U["reg"]["ab"][i].numpy()), None, None), (lab2rgb(L, U["pix2pix"]["ab"][i].numpy()), None, None),
                  (lab2rgb(L, P["ab"][i].numpy()), None, None), (P["spread"][i].numpy(), "viridis", (0, 0.06)), (err, "magma", (0, 0.12))]
        for j, (im, cm, lim) in enumerate(panels):
            ax[i, j].imshow(im, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1]); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(cols[j], fontsize=8)
    plt.tight_layout(pad=0.3); [fig.savefig(f"{out}/fig_colour.{e}", dpi=300) for e in ("png", "pdf")]; plt.close(fig)
    cols = ["Bicubic LR", "SR baseline", "ARC-IR SR", "True thermal"]; fig, ax = plt.subplots(2, 4, figsize=(8, 4.3), squeeze=False); c0, c1 = 40, 88
    for i, row in enumerate((0, 0)):                      # first tile; second row is a zoomed crop of the same tile
        ims = [F.interpolate(S["lr"][row:row + 1], size=hr, mode="bicubic", align_corners=False)[0, 0].numpy(), U["sr"]["sr"][row, 0].numpy(), P["sr"][row, 0].numpy(), S["hr"][row, 0].numpy()]
        lo, hi = np.percentile(ims[3], [1, 99])
        for j, im in enumerate(ims):
            ax[i, j].imshow(im[c0:c1, c0:c1] if i == 1 else im, cmap="inferno", vmin=lo, vmax=hi); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(cols[j], fontsize=9)
    plt.tight_layout(pad=0.3); [fig.savefig(f"{out}/fig_sr.{e}", dpi=300) for e in ("png", "pdf")]; log(f"saved figures to {out}")


if __name__ == "__main__":
    main()

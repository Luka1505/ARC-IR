"""FAIR BASELINES on the same tiles, the same inputs and the same metrics as ARC-IR.
  --method reg      : L1 regression colorizer (small U-Net) on bicubic thermal                          -> chroma
  --method pix2pix  : the same U-Net trained as a conditional GAN (L1*100 + LSGAN, PatchGAN critic)    -> chroma
  --method sr       : learned residual super-resolution (L1) on the thermal image                       -> SR PSNR / SSIM
  --inputs thermal  : bicubic thermal + metadata only (what a published method sees)
  --inputs full     : + terrain, absolute temperature and other dates (exactly what ARC-IR sees)  <- the fair comparison
Metrics (same as ARC-IR): chroma MAE, class-flip, label error of the frozen segmenter reading the predicted colour, selective risk when the
segmenter's own confidence is used to abstain (the standard rule a baseline can use), per test region.
Run (landsat, same flags as ARC-IR):
  python -m v1_baselines.run_fair --data landsat --regions data_tools/regions_smoke.yaml --steps 3000 --bs 8 --soft --terrain --dates 3 --abs-temp --color-target fixed --method reg --inputs full --seed 0 --out runs/b_reg_full_s0
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common.utils import add_common_args, setup, get_data, train_seg, to_dev, batch, cosine_lr, Timer, save_json
from common import metrics as M
from common.models import conv


class UNet(nn.Module):
    def __init__(s, cin, cout, c=48):
        super().__init__()
        s.e1 = nn.Sequential(conv(cin, c), conv(c, c)); s.e2 = nn.Sequential(conv(c, 2 * c), conv(2 * c, 2 * c)); s.e3 = nn.Sequential(conv(2 * c, 4 * c), conv(4 * c, 4 * c))
        s.d2 = nn.Sequential(conv(6 * c, 2 * c), conv(2 * c, 2 * c)); s.d1 = nn.Sequential(conv(3 * c, c), conv(c, c)); s.out = nn.Conv2d(c, cout, 3, padding=1)

    def forward(s, x):
        H, W = x.shape[-2:]; p = lambda t: F.pad(t, (0, (-t.shape[-1]) % 4, 0, (-t.shape[-2]) % 4), mode="replicate")
        x = p(x); a = s.e1(x); b = s.e2(F.avg_pool2d(a, 2)); c = s.e3(F.avg_pool2d(b, 2))
        b = s.d2(torch.cat([b, F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)], 1))
        a = s.d1(torch.cat([a, F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)], 1))
        return s.out(a)[..., :H, :W]


class Critic(nn.Module):                                   # PatchGAN on (thermal, chroma)
    def __init__(s, cin=3, c=48):
        super().__init__()
        s.n = nn.Sequential(nn.Conv2d(cin, c, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(c, 2 * c, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(4 * c, 1, 3, padding=1))

    def forward(s, x):
        return s.n(x)


def make_input(b, hr_size, inputs):
    up = lambda t: F.interpolate(t.float(), size=hr_size, mode="bicubic", align_corners=False)
    xs = [up(b["lr"])]
    if inputs == "full" and "aux" in b:
        xs.append(up(b["aux"]))
    xs.append(b["m"][:, :, None, None].expand(-1, -1, *hr_size))
    return torch.cat(xs, 1)


def fwd(net, x, method):
    y = net(x)
    return x[:, :1] + y if method == "sr" else y


@torch.no_grad()
def predict(net, D, dev, method, inputs, bs=16):
    net.eval(); out = []
    for i in range(0, len(D["hr"]), bs):
        b = {k: D[k][i:i + bs].to(dev) for k in ["lr", "m"] + (["aux"] if "aux" in D else [])}
        out.append(fwd(net, make_input(b, D["hr"].shape[-2:], inputs), method).cpu())
    return torch.cat(out)


@torch.no_grad()
def seg_read(seg, D, ab, dev, bs=32):
    """class the frozen segmenter reads from (L, colour) and its confidence"""
    cls, conf = [], []
    for i in range(0, D["L"].shape[0], bs):
        p = F.softmax(seg(D["L"][i:i + bs].to(dev), ab[i:i + bs].to(dev)), 1); m, c = p.max(1)
        cls.append(c.cpu().to(torch.uint8)); conf.append(m.cpu())
    return torch.cat(cls), torch.cat(conf)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--method", default="reg", choices=["reg", "pix2pix", "sr"]); ap.add_argument("--inputs", default="full", choices=["thermal", "full"])
    ap.add_argument("--width", type=int, default=48); ap.add_argument("--l1", type=float, default=100.0, help="pix2pix L1 weight")
    a = ap.parse_args(); dev, out, log = setup(a, f"b_{a.method}_{a.inputs}_s{a.seed}")
    tr, ca, te, info = get_data(a, log); seg = train_seg(tr, info, dev); hr = tr["hr"].shape[-2:]
    b0 = {k: tr[k][:2] for k in ["lr", "m"] + (["aux"] if "aux" in tr else [])}; cin = make_input(b0, hr, a.inputs).shape[1]
    cout = 1 if a.method == "sr" else 2
    net = UNet(cin, cout, a.width).to(dev); params = list(net.parameters()); opt = torch.optim.AdamW(params, lr=a.lr / 4 if a.method != "sr" else a.lr / 4)
    if a.method == "pix2pix":
        D = Critic(3, a.width).to(dev); optd = torch.optim.AdamW(D.parameters(), lr=a.lr / 4, betas=(0.5, 0.999))
    tm = Timer(a.steps, log); log(f"training {a.method} | inputs {a.inputs} ({cin} channels) | {sum(p.numel() for p in params) / 1e6:.2f} M parameters")
    N = len(tr["hr"])
    for step in range(a.steps):
        cosine_lr(opt, a.lr / 4, step, a.steps); i = torch.randint(0, N, (a.bs,)); b = to_dev(batch(tr, i), dev)
        x = make_input(b, hr, a.inputs); y = fwd(net, x, a.method)
        if a.method == "sr":
            loss = F.l1_loss(y, b["hr"])
        elif a.method == "reg":
            loss = F.l1_loss(y, b["ab"])
        else:
            real, fake = torch.cat([b["hr"], b["ab"]], 1), torch.cat([b["hr"], y.detach()], 1)
            ld = 0.5 * (F.mse_loss(D(real), torch.ones_like(D(real))) + F.mse_loss(D(fake), torch.zeros_like(D(fake))))
            optd.zero_grad(); ld.backward(); optd.step()
            lg = F.mse_loss(D(torch.cat([b["hr"], y], 1)), torch.ones_like(D(real)))
            loss = lg + a.l1 * F.l1_loss(y, b["ab"])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tm.tick(step)
        if step % 200 == 0 or step == a.steps - 1: log(f"    step {step:5d} loss={loss.item():.4f}")
    P = predict(net, te, dev, a.method, a.inputs); res = {"method": a.method, "inputs": a.inputs, "seed": a.seed}
    if a.method == "sr":
        bic = F.interpolate(te["lr"], size=te["hr"].shape[-2:], mode="bicubic", align_corners=False)
        res.update(sr_psnr=M.psnr(P, te["hr"]), sr_ssim=M.ssim(P, te["hr"]), bicubic_psnr=M.psnr(bic, te["hr"]), bicubic_ssim=M.ssim(bic, te["hr"]))
        log(f"test: SR PSNR {res['sr_psnr']:.2f} SSIM {res['sr_ssim']:.4f} | bicubic PSNR {res['bicubic_psnr']:.2f} SSIM {res['bicubic_ssim']:.4f}")
    else:
        names = info["names"]; lab = te["lab"]; ft = M.class_flips(seg, te, P, dev); cls, conf = seg_read(seg, te, P, dev); cls_true, _ = seg_read(seg, te, te["ab"], dev)
        err = (cls != lab).float().flatten().numpy(); conf = conf.flatten().numpy()
        res.update(chroma_mae=float((P - te["ab"]).abs().mean()), flip_full_coverage=float(ft.mean()), label_error_seg_read=float(err.mean()),
                   label_error_seg_on_true_colour=float((cls_true != lab).float().mean()))
        g, r = M.risk_coverage(-conf, err); res["selective_label_error_seg_confidence"] = {f"coverage_{int(c * 100)}": float(r[int(c * 96) - 1]) for c in (0.5, 0.7, 0.9)}
        log(f"test: chroma MAE {res['chroma_mae']:.4f} | class-flip {res['flip_full_coverage']:.3f} | label error (segmenter reads our colour) {res['label_error_seg_read']:.3f} "
            f"| (segmenter on the TRUE colour {res['label_error_seg_on_true_colour']:.3f})")
        log(f"      label error when abstaining with the segmenter's confidence: " + ", ".join(f"{k.split('_')[1]}% coverage {v:.3f}" for k, v in res["selective_label_error_seg_confidence"].items()))
        rn = info.get("reg_names", {}).get("test", []); reg = te["reg"].numpy() if "reg" in te else np.zeros(len(lab), int); npx = lab.shape[-1] * lab.shape[-2]
        rp = np.repeat(reg, npx); fl = ft; per = {}
        log(f"      {'region':24s} {'class-flip':>10s} {'label error':>12s}")
        for r_ in np.unique(rp):
            m = rp == r_; nm = rn[r_] if r_ < len(rn) else f"test{r_}"; per[nm] = {"class_flip": float(fl[m].mean()), "label_error": float(err[m].mean())}
            log(f"      {nm:24s} {per[nm]['class_flip']:10.3f} {per[nm]['label_error']:12.3f}")
        res["per_region"] = per
    torch.save({"state": net.state_dict(), "args": vars(a)}, f"{out}/net.pt"); save_json(res, f"{out}/results.json"); log(f"saved {out}/results.json")


if __name__ == "__main__":
    main()

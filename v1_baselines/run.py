"""VERSION 1 - baselines. Answers: what number does ARC-IR have to beat?
  * bicubic upscaling (SR floor)
  * regression colorizer on bicubic thermal, no metadata   (Pix2Pix-like stand-in)
  * regression colorizer on bicubic thermal + metadata     (the control reviewers will ask for)
Real published baselines (pix2pix, PearlGAN, MFmamba, HAT) are run from their own repos on the same tiles; see README.
Run:  python -m v1_baselines.run --data synthetic --steps 300
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
from common.utils import add_common_args, setup, get_data, train_seg, to_dev, batch, cosine_lr, Timer, save_json
from common.models import conv
from common import metrics as M


class RegColor(nn.Module):
    def __init__(s, mdim, use_meta):
        super().__init__()
        s.use_meta = use_meta
        s.net = nn.Sequential(conv(1 + (mdim if use_meta else 0), 32), conv(32, 32), conv(32, 32), nn.Conv2d(32, 2, 3, padding=1))

    def forward(s, up, m):
        x = up
        if s.use_meta:
            x = torch.cat([up, m[:, :, None, None].expand(-1, -1, up.shape[2], up.shape[3])], 1)
        return s.net(x)


def main():
    ap = add_common_args(argparse.ArgumentParser()); a = ap.parse_args()
    dev, out, log = setup(a, "v1_baselines")
    tr, ca, te, info = get_data(a, log)
    seg = train_seg(tr, info, dev)
    up = lambda D, i=slice(None): F.interpolate(D["lr"][i], size=D["hr"].shape[-2:], mode="bicubic", align_corners=False)
    res = {"bicubic": {"sr_psnr": M.psnr(up(te), te["hr"]), "sr_ssim": M.ssim(up(te), te["hr"])}}
    log(f"bicubic SR: PSNR {res['bicubic']['sr_psnr']:.2f}  SSIM {res['bicubic']['sr_ssim']:.4f}")
    for name, meta in [("reg_no_metadata", False), ("reg_with_metadata (control)", True)]:
        torch.manual_seed(a.seed)
        net = RegColor(info["mdim"], meta).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr); tm = Timer(a.steps, log)
        log(f"training {name}")
        for step in range(a.steps):
            cosine_lr(opt, a.lr, step, a.steps)
            i = torch.randint(0, len(tr["hr"]), (a.bs,)); b = to_dev(batch(tr, i), dev)
            u = F.interpolate(b["lr"], size=b["hr"].shape[-2:], mode="bicubic", align_corners=False)
            loss = F.l1_loss(net(u, b["m"]), b["ab"])
            opt.zero_grad(); loss.backward(); opt.step(); tm.tick(step)
            if step % 100 == 0: log(f"    step {step:5d} l1={loss.item():.4f}")
        net.eval(); abs_ = []
        with torch.no_grad():
            for i in range(0, len(te["hr"]), 16):
                sl = slice(i, i + 16)
                abs_.append(net(up(te, sl).to(dev), te["m"][sl].to(dev)).cpu())
        ab = torch.cat(abs_); ft = M.class_flips(seg, te, ab, dev)
        res[name] = {"chroma_mae": float((ab - te["ab"]).abs().mean()), "flip_full_coverage": float(ft.mean())}
        log(f"  {name}: chroma MAE {res[name]['chroma_mae']:.4f}  class-flip {res[name]['flip_full_coverage']:.3f}")
    save_json(res, f"{out}/results.json"); log(f"saved {out}/results.json")


if __name__ == "__main__":
    main()

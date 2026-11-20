"""Short test of the ARC-IR structure branch (SR only): which learning rate / edge weight works? Loads a capped subset of tiles once, trains each config for --steps,
and prints test PSNR next to plain bilinear and bicubic upsampling and next to the SR baseline's 35.35 dB (full data, 30k steps).
Run:  python -m v6_final.stage1_test --data landsat --tiles-dir dataset_big --regions data_tools/regions_big.yaml --soft --terrain --dates 3 --abs-temp --color-target fixed --max-tiles 10000 --steps 3000 --bs 8 --out runs/stage1_test
"""
import argparse, time, torch, torch.nn.functional as F
from common.utils import add_common_args, setup, get_data, to_dev, batch, cosine_lr
from common.models import build_structure, edges
from common import metrics as M

CONFIGS = [(2e-3, 0.05), (5e-4, 0.05), (5e-4, 0.0), (2e-3, 0.0), (1e-3, 0.0)]       # (learning rate, edge weight)


def main():
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--width-mult", type=int, default=4); a = ap.parse_args()
    dev, out, log = setup(a, "stage1_test"); tr, ca, te, info = get_data(a, log); N = len(tr["hr"]); n = min(3000, len(te["hr"]))
    lr, hr = te["lr"][:n], te["hr"][:n]
    log(f"reference on {n} test tiles: bilinear {M.psnr(F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False), hr):.2f} dB | bicubic {M.psnr(F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False), hr):.2f} dB")
    for lr0, we in CONFIGS:
        torch.manual_seed(0); st = build_structure("small", a.scale, None, a.width_mult).to(dev); opt = torch.optim.AdamW(st.parameters(), lr=lr0, weight_decay=1e-4); t0 = time.time()
        for step in range(a.steps):
            cosine_lr(opt, lr0, step, a.steps); b = to_dev(batch(tr, torch.randint(0, N, (a.bs,))), dev); _, sr = st(b["lr"])
            l1, le = F.l1_loss(sr, b["hr"]), F.l1_loss(edges(sr), edges(b["L"])); loss = l1 + we * le
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0); opt.step()
        st.eval()
        with torch.no_grad(): sr = torch.cat([st(lr[i:i + 64].to(dev))[1].cpu() for i in range(0, n, 64)])
        log(f"lr {lr0:g} edge {we:g}: final train l1 {l1.item():.4f} | test PSNR {M.psnr(sr, hr):.2f} dB SSIM {M.ssim(sr, hr):.4f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

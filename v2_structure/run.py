"""VERSION 2 - structure branch only (super-resolution). Answers: does our SR beat bicubic / the pretrained backbone?
  --structure small          trains the small stand-in from scratch
  --structure hat --weights  fine-tunes the pretrained HAT (needs basicsr + weights; untested here)
Saves runs/v2_structure/structure.pt for v3.
Run:  python -m v2_structure.run --data synthetic --steps 400
"""
import argparse, torch, torch.nn.functional as F
from common.utils import add_common_args, setup, get_data, to_dev, batch, cosine_lr, Timer, save_json
from common.models import build_structure, edges
from common import metrics as M


def main():
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--w-edge", type=float, default=0.3); a = ap.parse_args()
    dev, out, log = setup(a, "v2_structure")
    tr, ca, te, info = get_data(a, log)
    st = build_structure(a.structure, a.scale, a.weights).to(dev)
    log(f"structure '{a.structure}': {sum(p.numel() for p in st.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(st.parameters(), lr=a.lr, weight_decay=1e-4); tm = Timer(a.steps, log)
    for step in range(a.steps):
        cosine_lr(opt, a.lr, step, a.steps)
        i = torch.randint(0, len(tr["hr"]), (a.bs,)); b = to_dev(batch(tr, i), dev)
        _, sr = st(b["lr"])
        l1, le = F.l1_loss(sr, b["hr"]), F.l1_loss(edges(sr), edges(b["L"]))
        loss = l1 + a.w_edge * le
        opt.zero_grad(); loss.backward(); opt.step(); tm.tick(step)
        if step % 100 == 0 or step == a.steps - 1: log(f"    step {step:5d} l1={l1.item():.4f} edge={le.item():.3f}")
    st.eval(); srs = []
    with torch.no_grad():
        for i in range(0, len(te["hr"]), 16): srs.append(st(te["lr"][i:i + 16].to(dev))[1].cpu())
    sr = torch.cat(srs); bic = F.interpolate(te["lr"], size=te["hr"].shape[-2:], mode="bicubic", align_corners=False)
    res = {"sr_psnr": M.psnr(sr, te["hr"]), "sr_ssim": M.ssim(sr, te["hr"]), "bicubic_psnr": M.psnr(bic, te["hr"]), "bicubic_ssim": M.ssim(bic, te["hr"])}
    lp = M.lpips_rgbish(sr[:64].clamp(0, 1).to(dev), te["hr"][:64].to(dev))
    if lp is not None: res["sr_lpips"] = lp
    log(f"test: SR PSNR {res['sr_psnr']:.2f} SSIM {res['sr_ssim']:.4f}   bicubic PSNR {res['bicubic_psnr']:.2f} SSIM {res['bicubic_ssim']:.4f}")
    torch.save({"state": st.state_dict(), "structure": a.structure, "scale": a.scale}, f"{out}/structure.pt")
    save_json(res, f"{out}/results.json"); log(f"saved {out}/structure.pt and results.json")


if __name__ == "__main__":
    main()

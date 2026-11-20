"""VERSION 3 - add the material head + flow-matching chroma generator on top of the v2 structure branch.
Answers: does colour generation work, and how good is the colour at full coverage (no abstention yet)?
Run:  python -m v3_chroma.run --data synthetic --init runs/v2_structure/structure.pt --freeze-st --steps 600
Saves runs/v3_chroma/arc.pt for v4.
"""
import argparse, torch
from common.utils import add_common_args, setup, get_data, train_seg, save_json
from common.models import ARC
from common.train import train_arc
from common.evalx import predict, basic_report


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--init", default=None, help="v2 structure checkpoint"); ap.add_argument("--freeze-st", action="store_true")
    ap.add_argument("--no-meta", action="store_true"); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--w-edge", type=float, default=0.05, help="edge-consistency weight (only used when the structure branch is not frozen)"); a = ap.parse_args()
    dev, out, log = setup(a, "v3_chroma")
    tr, ca, te, info = get_data(a, log)
    seg = train_seg(tr, info, dev)
    model = ARC("arc", info, a.structure, a.scale, a.weights, aux_ch=info["aux_ch"]).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev); model.st.load_state_dict(ck["state"]); log(f"loaded structure from {a.init}")
    use_meta = not a.no_meta
    train_arc(model, tr, a, dev, log, w_inv=0.0, use_meta=use_meta, freeze_st=a.freeze_st, w_edge=a.w_edge)
    P = predict(model, te, dev, use_meta, S=a.samples)
    r, _ = basic_report(P, te, seg, dev, info["names"])
    log(f"test: SR PSNR {r['sr_psnr']:.2f} SSIM {r['sr_ssim']:.4f} | chroma MAE {r['chroma_mae']:.4f} | class-flip {r['flip_full_coverage']:.3f} | cls acc {r['cls_acc']:.3f}")
    torch.save({"state": model.state_dict(), "args": vars(a), "info": info, "use_meta": use_meta}, f"{out}/arc.pt")
    save_json(r, f"{out}/results.json"); log(f"saved {out}/arc.pt and results.json")


if __name__ == "__main__":
    main()

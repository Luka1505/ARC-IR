"""VERSION 5 - full ARC-IR, trained jointly, plus the ablation table the paper needs.
Variants: Reg (no metadata) | Reg + metadata (control) | ARC-IR full | w/o invariance | unfactorised | w/o metadata
Run:  python -m v5_full.run --data synthetic --steps 600 --seeds 1        (add --init runs/v2_structure/structure.pt to start from v2)
"""
import argparse, time, numpy as np, torch
from common.utils import add_common_args, setup, get_data, train_seg, save_json
from common.models import ARC
from common.train import train_arc
from common.evalx import predict, basic_report, abstention_report

VARIANTS = {
    "Reg (single estimate, no metadata)": dict(mode="reg", meta=False),
    "Reg + metadata (control)": dict(mode="reg", meta=True),
    "ARC-IR (full)": dict(mode="arc", meta=True, w_inv=1.0),
    "ARC-IR w/o invariance loss": dict(mode="arc", meta=True, w_inv=0.0),
    "Unfactorized, equal size": dict(mode="unfact", meta=True),
    "ARC-IR w/o metadata": dict(mode="arc", meta=False, w_inv=1.0),
}


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--only", default=""); ap.add_argument("--init", default=None); ap.add_argument("--samples", type=int, default=8); ap.add_argument("--w-edge", type=float, default=0.05)
    a = ap.parse_args(); dev, out, log = setup(a, "v5_full")
    tr, ca, te, info = get_data(a, log); seg = train_seg(tr, info, dev); allres = {}
    for seed in range(a.seeds):
        for name, cfg in VARIANTS.items():
            if a.only and a.only not in name: continue
            torch.manual_seed(a.seed + seed); np.random.seed(a.seed + seed)
            log(f"[seed {seed}] {name}")
            model = ARC(cfg["mode"], info, a.structure, a.scale, a.weights, aux_ch=info["aux_ch"]).to(dev)
            if a.init:
                model.st.load_state_dict(torch.load(a.init, map_location=dev)["state"])
            train_arc(model, tr, a, dev, log, w_inv=cfg.get("w_inv", 0.0), use_meta=cfg["meta"], w_edge=a.w_edge)
            Pc, Pt = predict(model, ca, dev, cfg["meta"], S=a.samples), predict(model, te, dev, cfg["meta"], S=a.samples)
            _, fc = basic_report(Pc, ca, seg, dev, info["names"]); r, ft = basic_report(Pt, te, seg, dev, info["names"])
            r["abstention"] = abstention_report(Pc, fc, Pt, ft, te["lab"], info["names"], a.alpha, cfg["mode"] != "reg", te)
            key = "ambiguity" if cfg["mode"] != "reg" else "entropy"
            # counterfactual stability: same place, other date -> same predicted class?
            ok = te["pair"] >= 0
            r["cf_agree_across_dates"] = float((Pt["pred"][ok] == Pt["pred"][te["pair"][ok]]).float().mean()) if ok.any() else float("nan")
            r["params"] = sum(p.numel() for p in model.parameters())
            t0 = time.time(); predict(model, {k: v[:1] for k, v in te.items()}, dev, cfg["meta"], S=a.samples); r["ms_per_tile"] = (time.time() - t0) * 1000
            allres.setdefault(name, []).append(r); d = r["abstention"][key]
            log(f"    flip@full {r['flip_full_coverage']:.3f} | AURC {d['aurc']:.3f} | cov@{a.alpha} {d['coverage_at_alpha']:.2f} risk {d['test_risk_at_alpha']:.3f} "
                f"| SR {r['sr_psnr']:.2f}dB | cf-agree {r['cf_agree_across_dates']:.3f} | {r['ms_per_tile']:.0f} ms/tile")
    save_json(allres, f"{out}/results.json"); log(f"saved {out}/results.json")


if __name__ == "__main__":
    main()

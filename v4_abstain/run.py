"""VERSION 4 - commit-or-abstain. No training: takes the v3 model, learns the ambiguity score on the CALIBRATION regions,
sets the threshold so class-flip rate of committed pixels <= alpha, and reports risk-coverage on the TEST regions.
Run:  python -m v4_abstain.run --data synthetic --ckpt runs/v3_chroma/arc.pt --alpha 0.05
Saves results.json, risk-coverage plot and ambiguity maps for a few test tiles.
"""
import argparse, numpy as np, torch
from common.utils import add_common_args, setup, get_data, train_seg, save_json
from common.models import ARC
from common.evalx import predict, basic_report, abstention_report, region_report
from common import metrics as M


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--ckpt", default="runs/v3_chroma/arc.pt"); ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--region-q", type=float, default=0.2, help="region-aware calibration: quantile over per-region thresholds (lower = safer)")
    a = ap.parse_args(); dev, out, log = setup(a, "v4_abstain")
    ck = torch.load(a.ckpt, map_location=dev); ca_args = ck["args"]
    a.dates, a.terrain, a.soft, a.abs_temp, a.color_target = ca_args.get("dates", 1), ca_args.get("terrain", False), ca_args.get("soft", False), ca_args.get("abs_temp", False), ca_args.get("color_target", "stretch")   # inputs must match training
    tr, ca, te, info = get_data(a, log); seg = train_seg(tr, info, dev)
    model = ARC("arc", info, ca_args["structure"], ca_args["scale"], ca_args.get("weights"), aux_ch=info["aux_ch"]).to(dev); model.load_state_dict(ck["state"]); model.eval()
    use_meta = ck["use_meta"]
    Pc, Pt = predict(model, ca, dev, use_meta, S=a.samples), predict(model, te, dev, use_meta, S=a.samples)
    _, fc = basic_report(Pc, ca, seg, dev, info["names"]); r, ft = basic_report(Pt, te, seg, dev, info["names"])
    ab, sc = abstention_report(Pc, fc, Pt, ft, te["lab"], info["names"], a.alpha, True, te, return_scores=True)
    for k in ["ambiguity", "entropy", "spread", "random"]:
        d = ab[k]; log(f"  {k:9s} AURC {d['aurc']:.3f} | coverage@alpha {d['coverage_at_alpha']:.2f} | test risk {d['test_risk_at_alpha']:.3f}")
    log("commit rate by class (ambiguity): " + ", ".join(f"{k} {v:.2f}" for k, v in ab["ambiguity"]["commit_rate_by_class"].items()))
    # region-aware calibration (does the alpha guarantee survive NEW regions?)
    npx = lambda D: D["lab"].shape[-1] * D["lab"].shape[-2]
    rep = lambda D: np.repeat(D["reg"].numpy(), npx(D))
    rr = region_report(sc["ambiguity"][0], sc["ambiguity"][1], fc, ft, rep(ca), rep(te), Pc["pred"].flatten().numpy(), Pt["pred"].flatten().numpy(),
                       a.alpha, a.region_q, info["names"], info["reg_names"]["test"])
    log(f"  region-aware calibration on {rr['n_cal_regions']} calibration regions (alpha {a.alpha}, q {a.region_q}):")
    for k in ["pooled", "region_quantile", "region_quantile_mondrian"]:
        d = rr[k]; rk = "n/a" if d["risk"] is None else f"{d['risk']:.3f}"; wr = "n/a" if d["worst_region_risk"] is None else f"{d['worst_region_risk']:.3f}"
        ok = "n/a" if d["frac_test_regions_ok"] is None else f"{d['frac_test_regions_ok']:.2f}"
        log(f"    {k:26s} coverage {d['coverage']:.2f} | risk {rk} | worst test region {wr} | test regions within alpha {ok}")
    res = {"full_coverage": r, "abstention": ab, "region_calibration": rr, "alpha": a.alpha}; save_json(res, f"{out}/results.json")
    # ambiguity maps of the first 8 test tiles (committed = True where the ambiguity score is under the calibrated threshold)
    w = ab["amb_weights"]; spc, enc = Pc["spread"].flatten().numpy(), Pc["ent"].flatten().numpy()
    amb = (w[0] * M.zfit(Pt["spread"].numpy(), spc) + w[1] * M.zfit(Pt["ent"].numpy(), enc))
    np.savez_compressed(f"{out}/ambiguity_maps.npz", ambiguity=amb[:8], committed=(amb[:8] <= ab["ambiguity"]["threshold"] - 0), L=te["L"][:8].numpy(),
                        ab_pred=Pt["ab"][:8].numpy(), ab_true=te["ab"][:8].numpy())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        g = np.linspace(0.05, 1, 96)[::8]; plt.figure(figsize=(5, 3.6))
        for k in ["ambiguity", "entropy", "spread", "random"]: plt.plot(g, ab[k]["curve"], marker="o", ms=3, label=k)
        plt.xlabel("coverage"); plt.ylabel("class-flip rate on coloured pixels"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/risk_coverage.png", dpi=130)
    except Exception as e:
        log(f"(plot skipped: {e})")
    log(f"saved {out}/results.json")


if __name__ == "__main__":
    main()

"""Turns the final runs into the paper's numbers. Needs runs/final_<method>_s<seed>/ for the methods that exist.
  1. main table (mean +- std over seeds), with and without 8-fold TTA, parameter counts, ms/tile
  2. hybrid head vs flow-mean ablation, task-driven variant check (MAE worse by > 1% than hybrid -> FLAG)
  3. uncertainty: ARC-IR flow spread / class-head entropy vs regression ensemble spread (3 seeds) vs learned-variance head vs segmenter confidence:
     selective risk (class-flip and label error) at 50/70/90 % coverage, AURC, and the calibrated commit at --alpha
  4. conformal colour intervals: radius = quantile of |point - truth| / spread on the CALIBRATION regions; compared with a constant radius at equal test coverage;
     per-test-region coverage is reported (calibrated per region set, NOT an absolute guarantee under region shift)
Pixels are sub-sampled on a regular 4 x 4 grid for the quantiles and risk curves (25 M pixels per split); everything is deterministic.
Run:  python -m v6_final.aggregate --data landsat --tiles-dir dataset_big --regions data_tools/regions_big.yaml --terrain --dates 3 --abs-temp --soft --color-target fixed --judge-dir runs/final_judge --out runs/final_summary
"""
import argparse, glob, json, os, numpy as np, torch, torch.nn.functional as F
from common.utils import add_common_args, setup, get_data, save_json
from common.judge import load_judges
from common import metrics as M

STR = 4; EPS = 1e-3; COV = (0.5, 0.7, 0.9)


def load(run, split, key):
    f = f"{run}/{split}_{key}.npy"; return np.load(f, mmap_mode="r") if os.path.exists(f) else None


@torch.no_grad()
def pixels(ab, spread, ent, D, seg, dev, bs=64):
    """sub-sampled per-pixel arrays: err (mean |ab - truth|), spread, flip, label error, segmenter confidence, class-head entropy, region id"""
    o = {k: [] for k in ["err", "spread", "flip", "lerr", "conf", "ent", "reg"]}
    for i in range(0, D["L"].shape[0], bs):
        L, t, lab = D["L"][i:i + bs].to(dev), D["ab"][i:i + bs].to(dev), D["lab"][i:i + bs].to(dev); p = torch.from_numpy(np.asarray(ab[i:i + bs], dtype=np.float32)).to(dev)
        pr = F.softmax(seg(L, p), 1); conf, cls = pr.max(1); ct = seg(L, t).argmax(1); s = lambda x: x[:, ::STR, ::STR].reshape(-1).cpu().numpy()
        o["err"].append(s((p - t).abs().mean(1))); o["flip"].append(s((cls != ct).float()).astype(bool)); o["lerr"].append(s((cls != lab).float()).astype(bool)); o["conf"].append(s(conf))
        n = s(lab).shape[0] // len(lab); o["reg"].append(np.repeat((D["reg"] if "reg" in D else torch.zeros(len(D["L"]), dtype=torch.long))[i:i + bs].numpy(), n).astype(np.int16))
        if spread is not None: o["spread"].append(s(torch.from_numpy(np.asarray(spread[i:i + bs], dtype=np.float32)).to(dev)))
        if ent is not None: o["ent"].append(s(torch.from_numpy(np.asarray(ent[i:i + bs], dtype=np.float32)).to(dev)))
    return {k: np.concatenate(v) for k, v in o.items() if v}


def scores_of(px, kind):
    sc = {"seg_confidence": -px["conf"]}
    if "ent" in px: sc["class_head_entropy"] = px["ent"]
    if "spread" in px: sc[kind] = px["spread"]
    sc["random"] = np.random.default_rng(0).random(len(px["err"])).astype(np.float32)
    return sc


def selective(pc, pt, kind, alpha):
    sc_c, sc_t = scores_of(pc, kind), scores_of(pt, kind); out = {}
    for name in sc_t:
        d = {}
        for e in ("flip", "lerr"):
            g, r = M.risk_coverage(sc_t[name], pt[e]); d[e] = {"aurc": float(r.mean()), **{f"risk@cov{int(c * 100)}": float(r[int(c * 96) - 1]) for c in COV}}
        tau = M.calibrate(sc_c[name], pc["flip"], alpha); cm = sc_t[name] <= tau
        d["calibrated_flip"] = {"alpha": alpha, "coverage": float(cm.mean()), "risk": float(pt["flip"][cm].mean()) if cm.any() else None}
        out[name] = d
    return out


def cq(s, delta):
    n = len(s); return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - delta)) / n), method="higher"))


def conformal(pc, pt, delta):
    out = {}
    for name, (sc, st, kind) in {"adaptive": (pc.get("spread"), pt.get("spread"), "spread"), "constant": (np.ones_like(pc["err"]), np.ones_like(pt["err"]), "const")}.items():
        if sc is None: continue
        q = cq(pc["err"] / (sc + EPS), delta)
        regs_c = np.unique(pc["reg"]); qr = [cq(pc["err"][pc["reg"] == r] / (sc[pc["reg"] == r] + EPS), delta) for r in regs_c if (pc["reg"] == r).sum() > 1000]
        q_reg = float(np.quantile(qr, 1 - 0.2, method="higher")) if len(qr) >= 3 else q                 # 80% of calibration regions at or above the target
        d = {}
        for tag, qq in (("pooled", q), ("region_quantile", q_reg)):
            cov = pt["err"] <= qq * (st + EPS); per = {int(r): float(cov[pt["reg"] == r].mean()) for r in np.unique(pt["reg"])}
            d[tag] = {"radius_scale": qq, "coverage": float(cov.mean()), "mean_radius": float((qq * (st + EPS)).mean()), "per_region_coverage_min": min(per.values()), "per_region_coverage_mean": float(np.mean(list(per.values()))),
                      "per_region_coverage_std": float(np.std(list(per.values()))), "per_region": per}
            d[tag]["constant_radius_at_equal_coverage"] = float(np.quantile(pt["err"], d[tag]["coverage"]))      # constant radius that reaches the same test coverage
        out[name] = d
    out["delta"] = delta
    return out


def analyse(ab_c, sp_c, en_c, ab_t, sp_t, en_t, ca, te, seg, dev, kind, alpha, delta):
    pc, pt = pixels(ab_c, sp_c, en_c, ca, seg, dev), pixels(ab_t, sp_t, en_t, te, seg, dev)
    return {"selective": selective(pc, pt, kind, alpha), "conformal": conformal(pc, pt, delta)}


class Ens:                                                    # lazy mean / std over seeds, chunk-indexable like a numpy array
    def __init__(s, arrs): s.a = arrs
    def mean(s, sl): return np.mean([np.asarray(x[sl], np.float32) for x in s.a], 0)
    def std(s, sl): return np.std([np.asarray(x[sl], np.float32) for x in s.a], 0).mean(1)


class V:                                                      # view with __getitem__
    def __init__(s, f): s.f = f
    def __getitem__(s, sl): return s.f(sl)


def ms(vals):
    v = [x for x in vals if x is not None]; return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "-"


def main():
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--judge-dir", required=True); ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--alpha", type=float, default=0.15); ap.add_argument("--delta", type=float, default=0.1)
    a = ap.parse_args(); dev, out, log = setup(a, "final_summary"); tr, ca, te, info = get_data(a, log); seg, _ = load_judges(a.judge_dir, tr, info, dev); del tr
    R = {}; runs = lambda m: [f"{a.runs_dir}/final_{m}_s{s}" for s in a.seeds if os.path.exists(f"{a.runs_dir}/final_{m}_s{s}/results.json")]
    methods = [m for m in ("arc", "arc_flowmean", "arc_task", "reg", "regvar", "pix2pix", "sr") if runs(m)]
    res = {m: [json.load(open(f"{r}/results.json")) for r in runs(m)] for m in methods}
    L = ["# Final results (mean ± std over seeds)\n", "## 1. Main table\n", "| method | seeds | params (M) | SR PSNR | SR SSIM | chroma MAE | class-flip | label err (judge reads colour) | ms/tile no-TTA | ms/tile TTA-8 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tt in ("tta8", "no_tta"):
        L.append(f"| **{tt}** | | | | | | | | | |")
        for m in methods:
            g = lambda k: ms([r[tt].get(k) for r in res[m]])
            L.append(f"| {m} | {len(res[m])} | {res[m][0]['params'] / 1e6:.2f} | {g('sr_psnr')} | {g('sr_ssim')} | {g('chroma_mae')} | {g('class_flip')} | {g('label_error_seg_read')} | {np.mean([r['ms_per_tile']['no_tta'] for r in res[m]]):.1f} | {np.mean([r['ms_per_tile']['tta8'] for r in res[m]]):.1f} |")
    if "arc" in res:
        L += ["\n## 2. Ablations\n", "| variant (TTA-8) | chroma MAE | class-flip | label err |", "|---|---|---|---|"]
        L.append(f"| ARC-IR hybrid point head | {ms([r['tta8']['chroma_mae'] for r in res['arc']])} | {ms([r['tta8']['class_flip'] for r in res['arc']])} | {ms([r['tta8']['label_error_seg_read'] for r in res['arc']])} |")
        if all("flow_mean" in r["tta8"] for r in res["arc"]):
            f = lambda k: ms([r["tta8"]["flow_mean"][k] for r in res["arc"]]); L.append(f"| ARC-IR flow-sample mean (same model) | {f('chroma_mae')} | {f('class_flip')} | {f('label_error_seg_read')} |")
        if "arc_task" in res:
            ma, mt = np.mean([r["tta8"]["chroma_mae"] for r in res["arc"]]), np.mean([r["tta8"]["chroma_mae"] for r in res["arc_task"]])
            L.append(f"| ARC-IR task-driven (2nd judge) | {ms([r['tta8']['chroma_mae'] for r in res['arc_task']])} | {ms([r['tta8']['class_flip'] for r in res['arc_task']])} | {ms([r['tta8']['label_error_seg_read'] for r in res['arc_task']])} |")
            L.append(f"\n**Task-driven check:** MAE {mt:.4f} vs hybrid {ma:.4f} = {100 * (mt / ma - 1):+.2f}%  ->  " + ("**FLAG: MAE worse by more than 1%, report as a trade-off, not as a gain**" if mt > 1.01 * ma else "within 1%"))
    U = {}                                                     # uncertainty + conformal, per method (seed 0 is the headline; all seeds saved in json)
    for m, kind in (("arc", "flow_spread"), ("arc_flowmean", "flow_spread"), ("arc_task", "flow_spread"), ("regvar", "learned_sigma"), ("reg", "none"), ("pix2pix", "none")):
        for r in runs(m):
            ab_c, ab_t = load(r, "cal", "ab"), load(r, "test", "ab")
            if ab_c is None: continue
            sp = (load(r, "cal", "spread"), load(r, "test", "spread")) if kind != "none" else (None, None); en = (load(r, "cal", "ent"), load(r, "test", "ent")) if m.startswith("arc") else (None, None)
            log(f"analysing {r}"); U[os.path.basename(r)] = analyse(ab_c, sp[0], en[0], ab_t, sp[1], en[1], ca, te, seg, dev, kind, a.alpha, a.delta)
    rr = [r for r in runs("reg") if load(r, "cal", "ab") is not None]
    if len(rr) >= 2:                                           # regression ensemble: mean colour, spread = std across seeds
        mk = lambda sp: (V(lambda sl: Ens([load(r, sp, "ab") for r in rr]).mean(sl)), V(lambda sl: Ens([load(r, sp, "ab") for r in rr]).std(sl)))
        (abc, spc), (abt, spt) = mk("cal"), mk("test"); log(f"analysing regression ensemble of {len(rr)} seeds")
        U["reg_ensemble"] = analyse(abc, spc, None, abt, spt, None, ca, te, seg, dev, "ensemble_spread", a.alpha, a.delta)
        with torch.no_grad():
            pt = torch.cat([torch.from_numpy(abt[slice(i, i + 256)]) for i in range(0, len(te["hr"]), 256)]); ft = M.class_flips(seg, te, pt, dev)
        U["reg_ensemble"]["colour"] = {"chroma_mae": float((pt - te["ab"]).abs().mean()), "class_flip": float(ft.mean())}
    L += ["\n## 3. Selective risk (class-flip) — lower is better; first seed of each method\n", f"| method | score | AURC | risk at 50% coverage | 70% | 90% | calibrated commit: coverage / risk (alpha={a.alpha}) |", "|---|---|---|---|---|---|---|"]
    for k, v in U.items():
        for sname, d in v["selective"].items():
            if k.endswith("_s1") or k.endswith("_s2"): continue
            f = d["flip"]; c = d["calibrated_flip"]; L.append(f"| {k} | {sname} | {f['aurc']:.4f} | {f['risk@cov50']:.4f} | {f['risk@cov70']:.4f} | {f['risk@cov90']:.4f} | {c['coverage']:.2f} / {c['risk'] if c['risk'] is None else round(c['risk'], 4)} |")
    L += [f"\n## 4. Conformal colour intervals (delta={a.delta}; radius scale from calibration regions)\n", "| method | interval | calibration | test coverage | worst-region coverage | coverage std over regions | mean radius | constant radius at same coverage |", "|---|---|---|---|---|---|---|---|"]
    for k, v in U.items():
        if k.endswith("_s1") or k.endswith("_s2"): continue
        for iname in ("adaptive", "constant"):
            if iname not in v["conformal"]: continue
            for cal in ("pooled", "region_quantile"):
                d = v["conformal"][iname][cal]; L.append(f"| {k} | {iname} | {cal} | {d['coverage']:.3f} | {d['per_region_coverage_min']:.3f} | {d['per_region_coverage_std']:.3f} | {d['mean_radius']:.4f} | {d['constant_radius_at_equal_coverage']:.4f} |")
    open(f"{out}/summary.md", "w").write("\n".join(L)); save_json({"results": res, "uncertainty": U}, f"{out}/summary.json"); log("\n".join(L)); log(f"saved {out}/summary.md and summary.json")


if __name__ == "__main__":
    main()

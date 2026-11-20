import time, numpy as np, torch, torch.nn.functional as F
from .models import sample_chroma
from . import metrics as M


@torch.no_grad()
def predict(model, D, dev, use_meta=True, S=8, steps=8, bs=4):
    out = {k: [] for k in ["ab", "spread", "ent", "pred", "sr", "pmax", "margin"]}
    model.eval()
    for i in range(0, len(D["hr"]), bs):
        lr, m = D["lr"][i:i + bs].to(dev), D["m"][i:i + bs].to(dev) * (1.0 if use_meta else 0.0)
        o = model.encode(lr, m, D["aux"][i:i + bs].to(dev) if "aux" in D else None)
        p = F.softmax(o["logits"], 1)
        ent = -(p * torch.log(p + 1e-8)).sum(1); top2 = p.topk(2, 1).values
        out["pmax"].append(top2[:, 0].cpu()); out["margin"].append((top2[:, 0] - top2[:, 1]).cpu())
        out["ent"].append(ent.cpu()); out["pred"].append(p.argmax(1).cpu()); out["sr"].append(o["sr"].cpu())
        if model.mode == "reg":
            out["ab"].append(model.head(o["cond"]).cpu()); out["spread"].append(torch.zeros_like(ent).cpu())
        else:
            x = sample_chroma(model, o, S, steps)
            out["ab"].append(x.mean(1).cpu()); out["spread"].append(x.std(1, unbiased=False).mean(1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def basic_report(P, D, seg, dev, names):
    """image quality + colour quality at full coverage (no abstention)"""
    K = len(names); r = {}
    r["sr_psnr"], r["sr_ssim"] = M.psnr(P["sr"], D["hr"]), M.ssim(P["sr"], D["hr"])
    bic = F.interpolate(D["lr"], size=D["hr"].shape[-2:], mode="bicubic", align_corners=False)
    r["bicubic_psnr"], r["bicubic_ssim"] = M.psnr(bic, D["hr"]), M.ssim(bic, D["hr"])
    lp = M.lpips_rgbish(P["sr"][:64].clamp(0, 1).to(dev), D["hr"][:64].to(dev))
    if lp is not None:
        r["sr_lpips"] = lp
    r["chroma_mae"] = float((P["ab"] - D["ab"]).abs().mean())
    ft = M.class_flips(seg, D, P["ab"], dev)
    r["flip_full_coverage"] = float(ft.mean())
    r["cls_acc"] = float((P["pred"] == D["lab"]).float().mean())
    r["label_error_head"] = 1.0 - r["cls_acc"]                       # class head vs WorldCover
    with torch.no_grad():                                            # frozen segmenter reading OUR colour vs WorldCover (same measure as the baselines)
        bad = 0
        for i in range(0, D["L"].shape[0], 32):
            bad += (seg(D["L"][i:i + 32].to(dev), P["ab"][i:i + 32].to(dev)).argmax(1).cpu() != D["lab"][i:i + 32]).sum().item()
    r["label_error_seg_read"] = bad / D["lab"].numel()
    lab = D["lab"].flatten().numpy()
    r["mean_spread_by_class"] = {names[k]: float(P["spread"].flatten().numpy()[lab == k].mean()) for k in range(K) if (lab == k).any()}
    return r, ft


def abstention_report(Pc, fc, Pt, ft, lab_test, names, alpha, has_spread, Dte, return_scores=False):
    """risk-coverage, calibrated commit rate and risk, per-class commit rate for each ambiguity score"""
    K = len(names); sc = {"entropy": (Pc["ent"].flatten().numpy(), Pt["ent"].flatten().numpy())}; res = {}
    if has_spread:
        sp_c, sp_t = Pc["spread"].flatten().numpy(), Pt["spread"].flatten().numpy()
        sc["spread"] = (sp_c, sp_t)
        w, b = M.fit_ambiguity(sp_c, sc["entropy"][0], fc)
        amb = lambda sp, en, spr, enr: w[0] * M.zfit(sp, spr) + w[1] * M.zfit(en, enr) + b
        sc["ambiguity"] = (amb(sp_c, sc["entropy"][0], sp_c, sc["entropy"][0]), amb(sp_t, sc["entropy"][1], sp_c, sc["entropy"][0]))
        res["amb_weights"] = [float(w[0]), float(w[1])]
    sc["random"] = (np.random.rand(len(fc)), np.random.rand(len(ft)))
    lab = lab_test.flatten().numpy()
    ab_err = (Pt["ab"] - Dte["ab"]).abs().mean(1).flatten().numpy()
    for name, (s_c, s_t) in sc.items():
        g, r = M.risk_coverage(s_t, ft); tau = M.calibrate(s_c, fc, alpha); commit = s_t <= tau
        d = {"aurc": float(r.mean()), "threshold": float(tau), "coverage_at_alpha": float(commit.mean()),
             "test_risk_at_alpha": float(ft[commit].mean()) if commit.any() else float("nan"),
             "curve": [float(x) for x in r[::8]]}
        if commit.any():
            d["chroma_mae_committed"] = float(ab_err[commit].mean())
        d["commit_rate_by_class"] = {names[k]: float(commit[lab == k].mean()) for k in range(K) if (lab == k).any()}
        res[name] = d
    return (res, sc) if return_scores else res


def region_report(s_c, s_t, fc, ft, reg_c, reg_t, pred_c, pred_t, alpha, q=0.2, names=None, reg_names_t=None, min_px=2000):
    """Region-aware calibration. Standard calibration pools all calibration pixels, which assumes a NEW region looks like the pooled
    calibration regions. Here every calibration REGION gets its own threshold (largest commit set with class-flip <= alpha in that region);
    the final threshold is the q-quantile over regions (q=0.2: 80% of regions are at or below alpha). 'mondrian' does the same per predicted class.
    Arrays are per pixel. Returns coverage and risk overall and per TEST region, plus the share of test regions with risk <= alpha."""
    from . import metrics as M
    import numpy as np
    regs_c, regs_t = np.unique(reg_c), np.unique(reg_t)
    tau_pooled = M.calibrate(s_c, fc, alpha)
    tau_r = np.array([M.calibrate(s_c[reg_c == r], fc[reg_c == r], alpha) for r in regs_c])
    tau_reg = float(np.quantile(tau_r, q, method="lower")) if len(tau_r) else -np.inf
    K = len(names) if names else int(max(pred_c.max(), pred_t.max()) + 1); tau_cls = np.full(K, tau_reg)
    for k in range(K):
        t = []
        for r in regs_c:
            m = (reg_c == r) & (pred_c == k)
            if m.sum() >= min_px: t.append(M.calibrate(s_c[m], fc[m], alpha))
        if len(t) >= 2: tau_cls[k] = float(np.quantile(np.array(t), q, method="lower"))
    out = {"alpha": alpha, "q": q, "n_cal_regions": int(len(regs_c)), "tau_per_cal_region": [float(x) for x in tau_r]}
    for name, commit in {"pooled": s_t <= tau_pooled, "region_quantile": s_t <= tau_reg, "region_quantile_mondrian": s_t <= tau_cls[pred_t]}.items():
        per = {}
        for r in regs_t:
            m = reg_t == r; c = commit & m
            per[reg_names_t[r] if reg_names_t else int(r)] = {"coverage": float(c.sum() / m.sum()), "risk": float(ft[c].mean()) if c.any() else None}
        risks = [v["risk"] for v in per.values() if v["risk"] is not None]
        out[name] = {"coverage": float(commit.mean()), "risk": float(ft[commit].mean()) if commit.any() else None,
                     "frac_test_regions_ok": float(np.mean([x <= alpha for x in risks])) if risks else None,
                     "worst_region_risk": float(max(risks)) if risks else None, "per_region": per}
    return out

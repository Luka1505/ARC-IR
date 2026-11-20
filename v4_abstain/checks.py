"""Two quick CHECKS on an already trained model (no training). Run them before building anything new into the model.

CHECK 1  graded commit.  Instead of "colour it or leave it grey", an ambiguous pixel falls back to a COARSER class (water | vegetation = forest+cropland |
         non-vegetation = bare+built). Question: at the same risk target alpha, how much more of the image can be coloured?
    python -m v4_abstain.checks --check graded --data landsat --regions data_tools/regions_smoke.yaml --ckpt runs/v3_all/arc.pt --alphas 0.05 0.10 0.15
    "risk" of a fine pixel = the frozen segmenter reads the predicted colour as a different CLASS than the true colour (as in v4).
    "risk" of a coarse pixel = the same, but only counting a change of GROUP (so coarse risk <= fine risk by construction).

CHECK 6  labelscore.  Which score predicts a WRONG class (against the labels)? Compares our ambiguity score, the class head's confidence and a score learned on label error.
    python -m v4_abstain.checks --check labelscore --data landsat --regions data_tools/regions_smoke.yaml --ckpt runs/v3_fixed/arc.pt

CHECK 5  classes.  Label shares per region and how well the frozen segmenter reads the TRUE colour per region (explains why class-flip can look perfect where the class is wrong).
    python -m v4_abstain.checks --check classes --data landsat --regions data_tools/regions_smoke.yaml --ckpt runs/v3_fixed/arc.pt

CHECK 4  regions.  Per-region table, Mondrian calibration, commit rate per class/group and the confusion of the worst region.
    python -m v4_abstain.checks --check regions --data landsat --regions data_tools/regions_smoke.yaml --ckpt runs/v3_fixed/arc.pt

CHECK 3  palette.  Does the colour generator USE the class the model already knows?  Compares the chroma error of the generator with simple "paint every pixel with the
         average colour of its class" images (class = the true one = the best a class-driven colour could do, or the predicted one).
    python -m v4_abstain.checks --check palette --data landsat --regions data_tools/regions_smoke.yaml --ckpt runs/v3_abs_all/arc.pt

CHECK 2  cross-date consistency.  Land cover rarely changes between two dates of the same place, so if the model says "forest" on one date and
         "cropland" on the other, at least one is wrong. Question: does the cross-date disagreement of a region track its real error, without labels?
    python -m v4_abstain.checks --check consistency --data landsat --regions data_tools/regions_smoke.yaml --ckpt runs/v3_soft_terrain/arc.pt
    Use a SINGLE-date checkpoint (trained without --dates): a model that already sees both dates would agree with itself by construction.
"""
import argparse, numpy as np, torch
from common.utils import add_common_args, setup, get_data, train_seg, save_json
from common.models import ARC
from common.evalx import predict, abstention_report
from common import metrics as M


def load(a, dev, log):
    ck = torch.load(a.ckpt, map_location=dev); ca_args = ck["args"]
    a.dates, a.terrain, a.soft, a.abs_temp, a.color_target = ca_args.get("dates", 1), ca_args.get("terrain", False), ca_args.get("soft", False), ca_args.get("abs_temp", False), ca_args.get("color_target", "stretch")   # inputs must match training
    tr, ca, te, info = get_data(a, log)
    for D in (tr, ca, te):
        if "reg" not in D:                                    # synthetic data has no region ids: one region per split
            D["reg"] = torch.zeros(len(D["hr"]), dtype=torch.long)
    model = ARC("arc", info, ca_args["structure"], ca_args["scale"], ca_args.get("weights"), aux_ch=info["aux_ch"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()
    return model, ck, tr, ca, te, info


# ----------------------------------------------------------------------------------------------------- check 1: graded commit
def group_map(spec, names):
    """'water|forest,cropland|bare,built' -> array class -> group id"""
    g = np.full(len(names), -1)
    for gi, grp in enumerate(spec.split("|")):
        for n in grp.split(","):
            g[names.index(n.strip())] = gi
    assert (g >= 0).all(), f"every class must be in a group: {spec}"
    return g


@torch.no_grad()
def fine_coarse_flips(seg, D, ab_pred, dev, gmap, bs=32):
    g = torch.tensor(gmap, device=dev); f, c = [], []
    for i in range(0, D["L"].shape[0], bs):
        L, ab, p = D["L"][i:i + bs].to(dev), D["ab"][i:i + bs].to(dev), ab_pred[i:i + bs].to(dev)
        x, y = seg(L, p).argmax(1), seg(L, ab).argmax(1)
        f.append((x != y).float().flatten().cpu()); c.append((g[x] != g[y]).float().flatten().cpu())
    return torch.cat(f).numpy(), torch.cat(c).numpy()


def thresholds(s, f, c, reg, alpha, q, by_region):
    """tau_fine: largest set coloured with the exact class at risk <= alpha.  tau_total: the set that ALSO colours the next-most-ambiguous pixels
    with their coarse group, as long as the mixed risk (fine error inside tau_fine, coarse error beyond it) stays <= alpha."""
    if not by_region:
        tau_f = M.calibrate(s, f, alpha); e = np.where(s <= tau_f, f, c); return tau_f, max(tau_f, M.calibrate(s, e, alpha))
    regs = np.unique(reg)
    tf = np.array([M.calibrate(s[reg == r], f[reg == r], alpha) for r in regs]); tau_f = float(np.quantile(tf, q, method="lower"))
    e = np.where(s <= tau_f, f, c)
    tt = np.array([M.calibrate(s[reg == r], e[reg == r], alpha) for r in regs]); return tau_f, max(tau_f, float(np.quantile(tt, q, method="lower")))


def evaluate(s, f, c, reg, tau_f, tau_t, alpha, reg_names):
    fine = s <= tau_f; coarse = (s > tau_f) & (s <= tau_t); commit = fine | coarse; err = np.where(fine, f, c)
    per = {}
    for r in np.unique(reg):
        m = reg == r; k = commit & m
        per[reg_names[r] if r < len(reg_names) else int(r)] = float(err[k].mean()) if k.any() else None
    risks = [v for v in per.values() if v is not None]
    return {"coverage": float(commit.mean()), "coverage_fine": float(fine.mean()), "coverage_coarse": float(coarse.mean()),
            "risk": float(err[commit].mean()) if commit.any() else None, "worst_region_risk": max(risks) if risks else None,
            "frac_test_regions_ok": float(np.mean([x <= alpha for x in risks])) if risks else None, "per_region_risk": per}


def check_graded(a, dev, out, log):
    model, ck, tr, ca, te, info = load(a, dev, log); names = info["names"]; gmap = group_map(a.groups, names)
    seg = train_seg(tr, info, dev); use_meta = ck["use_meta"]
    Pc, Pt = predict(model, ca, dev, use_meta, S=a.samples), predict(model, te, dev, use_meta, S=a.samples)
    fc, cc = fine_coarse_flips(seg, ca, Pc["ab"], dev, gmap); ft, ct = fine_coarse_flips(seg, te, Pt["ab"], dev, gmap)
    log(f"full coverage on test: class-flip {ft.mean():.3f} | coarse (group) flip {ct.mean():.3f}   groups: {a.groups}")
    log("  (if the two numbers are close, grouping classes cannot help; the gap is the room for graded commit)")
    gl_c, gl_t = gmap[ca["lab"].numpy()], gmap[te["lab"].numpy()]; share = np.bincount(gl_t.ravel(), minlength=gmap.max() + 1) / gl_t.size
    maj = int(np.bincount(gl_c.ravel(), minlength=gmap.max() + 1).argmax()); base_err = float((gl_t != maj).mean())
    head_err = float((gmap[Pt["pred"].numpy()] != gl_t).mean())
    log(f"trivial baseline: colour everything as the most common group in calibration -> group error on test {base_err:.3f} (test group shares {np.round(share, 2).tolist()});"
        f" the model's own group error from its class head is {head_err:.3f}")
    _, sc = abstention_report(Pc, fc, Pt, ft, te["lab"], names, 0.1, True, te, return_scores=True)
    s_c, s_t = sc[a.score]; npx = te["lab"].shape[-1] * te["lab"].shape[-2]
    rc, rt = np.repeat(ca["reg"].numpy(), ca["lab"].shape[-1] * ca["lab"].shape[-2]), np.repeat(te["reg"].numpy(), npx)
    rn = info.get("reg_names", {}).get("test", []); res = {"score": a.score, "full_flip": float(ft.mean()), "full_coarse_flip": float(ct.mean()), "baseline_group_error": base_err, "head_group_error": head_err, "alphas": {}}
    for al in a.alphas:
        log(f"alpha {al}:")
        res["alphas"][str(al)] = {}
        for tag, by_region in [("pooled", False), ("region-aware", True)]:
            tf, tt = thresholds(s_c, fc, cc, rc, al, a.region_q, by_region)
            for kind, tau in [("fine only", tf), ("graded   ", tt)]:
                d = evaluate(s_t, ft, ct, rt, tf, tau, al, rn); res["alphas"][str(al)][f"{tag}/{kind.strip()}"] = d
                rk = "n/a" if d["risk"] is None else f"{d['risk']:.3f}"; wr = "n/a" if d["worst_region_risk"] is None else f"{d['worst_region_risk']:.3f}"
                ok = "n/a" if d["frac_test_regions_ok"] is None else f"{d['frac_test_regions_ok']:.2f}"
                log(f"  {tag:12s} {kind} coverage {d['coverage']:.2f} (exact class {d['coverage_fine']:.2f} + group only {d['coverage_coarse']:.2f})"
                    f" | risk {rk} | worst region {wr} | regions within alpha {ok}")
    save_json(res, f"{out}/checks_graded.json"); log(f"saved {out}/checks_graded.json")


# ------------------------------------------------------------------------------------------------ check 2: cross-date consistency
@torch.no_grad()
def class_pred(model, D, dev, use_meta, bs=64):
    out = []
    for i in range(0, len(D["hr"]), bs):
        m = D["m"][i:i + bs].to(dev) * (1.0 if use_meta else 0.0)
        o = model.encode(D["lr"][i:i + bs].to(dev), m, D["aux"][i:i + bs].to(dev) if "aux" in D else None)
        out.append(o["logits"].argmax(1).to(torch.uint8).cpu())
    return torch.cat(out)


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def check_consistency(a, dev, out, log):
    model, ck, tr, ca, te, info = load(a, dev, log); names = info["names"]; use_meta = ck["use_meta"]
    if a.dates > 1:
        log("WARNING: this checkpoint already uses several dates as input, so it agrees with itself by construction. Use a single-date checkpoint.")
    rows, dis_all, err_all, cls_all = [], [], [], []
    for split, D in [("train", tr), ("calibration", ca), ("test", te)]:
        pred = class_pred(model, D, dev, use_meta); pair = D["pair"]; has = pair >= 0
        if not has.any():
            continue
        p1, p2, lab, reg = pred[has], pred[pair[has]], D["lab"][has], D["reg"][has]
        dis, err = (p1 != p2), (p1 != lab.to(p1.dtype))
        rn = info.get("reg_names", {}).get(split, [])
        for r in torch.unique(reg).tolist():
            m = reg == r
            rows.append({"region": rn[r] if r < len(rn) else f"{split}{r}", "split": split, "tiles": int(m.sum()),
                         "disagreement": float(dis[m].float().mean()), "class_error": float(err[m].float().mean())})
        if split != "train":
            dis_all.append(dis.flatten()); err_all.append(err.flatten()); cls_all.append(lab.flatten())
    log(f"{'region':24s} {'split':12s} {'tiles':>6s} {'cross-date disagreement':>24s} {'true class error':>17s}")
    for r in sorted(rows, key=lambda r: r["disagreement"]):
        log(f"{r['region']:24s} {r['split']:12s} {r['tiles']:6d} {r['disagreement']:24.3f} {r['class_error']:17.3f}")
    res = {"regions": rows}
    for tag, sel in [("all regions", rows), ("unseen regions only (calibration + test)", [r for r in rows if r["split"] != "train"])]:
        if len(sel) >= 4:
            d, e = np.array([r["disagreement"] for r in sel]), np.array([r["class_error"] for r in sel])
            a_, b_ = ranks(d), ranks(e); rho = float(((a_ - a_.mean()) * (b_ - b_.mean())).mean() / (a_.std() * b_.std())); res[f"spearman_{tag}"] = rho
            log(f"Spearman correlation, disagreement vs true error, {tag} ({len(sel)} regions): {rho:.2f}")
    if dis_all:
        dis, err, cls = torch.cat(dis_all).numpy(), torch.cat(err_all).numpy(), torch.cat(cls_all).numpy()
        pe_d, pe_a = float(err[dis].mean()) if dis.any() else float("nan"), float(err[~dis].mean()) if (~dis).any() else float("nan")
        log(f"pixel level (unseen regions): P(wrong | the two dates disagree) = {pe_d:.3f}   P(wrong | they agree) = {pe_a:.3f}   share of disagreeing pixels {dis.mean():.3f}")
        res["p_wrong_given_disagree"], res["p_wrong_given_agree"] = pe_d, pe_a
        log("disagreement by true class (seasonal classes like water and cropland may disagree legitimately): " +
            ", ".join(f"{names[k]} {dis[cls == k].mean():.2f}" for k in range(len(names)) if (cls == k).any()))
    log("reading: a high positive correlation (and P(wrong|disagree) well above P(wrong|agree)) means cross-date disagreement can estimate a NEW region's risk without labels.")
    save_json(res, f"{out}/checks_consistency.json"); log(f"saved {out}/checks_consistency.json")


# ------------------------------------------------------------------------------------------------ check 3: palette
def class_palette(D, K, bs=512):
    """mean (a,b) colour of every class over the training tiles"""
    s, n = torch.zeros(K, 2), torch.zeros(K)
    for i in range(0, len(D["ab"]), bs):
        a, l = D["ab"][i:i + bs].permute(1, 0, 2, 3), D["lab"][i:i + bs].long()
        for k in range(K):
            m = l == k
            if m.any(): s[k] += a[:, m].sum(1); n[k] += m.sum()
    return s / n.clamp(min=1)[:, None]


def check_palette(a, dev, out, log):
    model, ck, tr, ca, te, info = load(a, dev, log); names = info["names"]; K = len(names)
    seg = train_seg(tr, info, dev); pal = class_palette(tr, K); P = predict(model, te, dev, ck["use_meta"], S=a.samples)
    paint = lambda cls: pal[cls.long()].permute(0, 3, 1, 2)                       # (N,H,W) class -> (N,2,H,W) colour
    glob = tr["ab"].float().mean((0, 2, 3))[None, :, None, None].expand_as(te["ab"])
    rows = {"one global average colour": glob, "paint with the TRUE class colour (best a class can do)": paint(te["lab"]),
            "paint with the PREDICTED class colour": paint(P["pred"]), "generator (the model's own colour)": P["ab"],
            "half generator + half predicted-class colour": 0.5 * P["ab"] + 0.5 * paint(P["pred"])}
    res = {}
    log(f"class colours (a,b): " + ", ".join(f"{names[k]} {pal[k].tolist()[0]:+.2f},{pal[k].tolist()[1]:+.2f}" for k in range(K)))
    log(f"{'colour source':58s} {'chroma MAE':>11s} {'class-flip':>11s}")
    for name, x in rows.items():
        mae, fl = float((x - te["ab"]).abs().mean()), float(M.class_flips(seg, te, x.contiguous(), dev).mean()); res[name] = {"chroma_mae": mae, "class_flip": fl}
        log(f"{name:58s} {mae:11.4f} {fl:11.3f}")
    log("reading: if 'PREDICTED class colour' beats the generator, the generator is wasting the class information (a class-conditioned colour head should help).")
    log("        if even the TRUE class colour is not much better than the generator, colour error is mostly variation INSIDE a class (season, crops) and class accuracy is not the lever.")
    save_json(res, f"{out}/checks_palette.json"); log(f"saved {out}/checks_palette.json")


# ------------------------------------------------------------------------------------------------ check 4: regions (per-region view)
def check_regions(a, dev, out, log):
    from common.evalx import region_report
    model, ck, tr, ca, te, info = load(a, dev, log); names = info["names"]; gmap = group_map(a.groups, names); gnames = a.groups.split("|")
    seg = train_seg(tr, info, dev); use_meta = ck["use_meta"]
    Pc, Pt = predict(model, ca, dev, use_meta, S=a.samples), predict(model, te, dev, use_meta, S=a.samples)
    fc, _ = fine_coarse_flips(seg, ca, Pc["ab"], dev, gmap); ft, _ = fine_coarse_flips(seg, te, Pt["ab"], dev, gmap)
    _, sc = abstention_report(Pc, fc, Pt, ft, te["lab"], names, 0.1, True, te, return_scores=True)
    s_c, s_t = sc[a.score]; npx = te["lab"].shape[-1] * te["lab"].shape[-2]
    rc, rt = np.repeat(ca["reg"].numpy(), ca["lab"].shape[-1] * ca["lab"].shape[-2]), np.repeat(te["reg"].numpy(), npx)
    rn = info.get("reg_names", {}).get("test", []); rname = lambda r: rn[r] if r < len(rn) else f"test{r}"
    lab_t, pred_t = te["lab"].numpy().ravel(), Pt["pred"].numpy().ravel(); res = {"alphas": {}}
    log("PER TEST REGION at full coverage (colour everything): class-flip, class-head error, main class")
    log(f"  {'region':24s} {'class-flip':>10s} {'class error':>11s} {'main class':>11s}")
    full = {}
    for r in np.unique(rt):
        m = rt == r; full[int(r)] = (float(ft[m].mean()), float((pred_t[m] != lab_t[m]).mean()), names[int(np.bincount(lab_t[m]).argmax())])
        log(f"  {rname(r):24s} {full[int(r)][0]:10.3f} {full[int(r)][1]:11.3f} {full[int(r)][2]:>11s}")
    worst = max(full, key=lambda r: full[r][1]); m = rt == worst; K = len(names)
    log(f"confusion in the worst region by class error ({rname(worst)}): rows = TRUE class, columns = predicted (share of the true class's pixels)")
    log("  " + " " * 10 + "".join(f"{n:>9s}" for n in names))
    conf = {}
    for k in range(K):
        mk = m & (lab_t == k)
        if mk.sum() == 0: continue
        row = np.bincount(pred_t[mk], minlength=K) / mk.sum(); conf[names[k]] = row.tolist()
        log(f"  {names[k]:10s}" + "".join(f"{x:9.2f}" for x in row) + f"   ({int(mk.sum())} px)")
    res["worst_region"], res["confusion"] = rname(worst), conf
    fcl, ftl = (Pc["pred"].flatten().numpy() != ca["lab"].numpy().ravel()).astype(float), (pred_t != lab_t).astype(float)   # truth-based risk: wrong class vs the WorldCover label
    log(f"full coverage: class-flip {ft.mean():.3f} | label error (class head vs WorldCover) {ftl.mean():.3f}")
    for rk_name, (fcx, ftx) in {"class-flip": (fc, ft), "label error": (fcl, ftl)}.items():
        for al in (a.alphas if rk_name == "class-flip" else a.label_alphas):
            rr = region_report(s_c, s_t, fcx, ftx, rc, rt, Pc["pred"].flatten().numpy(), pred_t, al, a.region_q, names, rn)
            log(f"[risk = {rk_name}] alpha {al} (calibrated on {rr['n_cal_regions']} calibration regions, region quantile {a.region_q}):")
            for k in ["pooled", "region_quantile", "region_quantile_mondrian"]:
                d = rr[k]; rk = "n/a" if d["risk"] is None else f"{d['risk']:.3f}"; wr = "n/a" if d["worst_region_risk"] is None else f"{d['worst_region_risk']:.3f}"
                ok = "n/a" if d["frac_test_regions_ok"] is None else f"{d['frac_test_regions_ok']:.2f}"
                log(f"  {k:26s} coverage {d['coverage']:.2f} | risk {rk} | worst region {wr} | regions within alpha {ok}")
            d = rr["region_quantile_mondrian"]["per_region"]
            log("  per region (mondrian):  " + "  ".join(f"{k}: cov {v['coverage']:.2f} risk {'n/a' if v['risk'] is None else format(v['risk'], '.3f')}" for k, v in d.items()))
            tau = float(np.quantile(np.array(rr["tau_per_cal_region"]), a.region_q, method="lower")); commit = s_t <= tau
            gl = gmap[lab_t]; log("  commit rate by TRUE group (region-quantile): " + ", ".join(f"{gnames[g]} {commit[gl == g].mean():.2f}" for g in range(len(gnames)) if (gl == g).any()))
            res["alphas"][f"{rk_name}/{al}"] = rr
    # graded commit against the TRUE labels: fine = wrong class, coarse = wrong GROUP (water | forest,cropland | bare,built)
    cc_l = (gmap[Pc["pred"].flatten().numpy()] != gmap[ca["lab"].numpy().ravel()]).astype(float); ct_l = (gmap[pred_t] != gmap[lab_t]).astype(float)
    log(f"full coverage against labels: class error {ftl.mean():.3f} | group error {ct_l.mean():.3f}")
    for al in a.label_alphas:
        log(f"[graded, risk = label error] alpha {al}:")
        for tag, by_region in [("pooled", False), ("region-aware", True)]:
            tf, tt = thresholds(s_c, fcl, cc_l, rc, al, a.region_q, by_region)
            for kind, tau in [("fine only", tf), ("graded   ", tt)]:
                d = evaluate(s_t, ftl, ct_l, rt, tf, tau, al, rn); res["alphas"][f"graded-label/{al}/{tag}/{kind.strip()}"] = d
                rk = "n/a" if d["risk"] is None else f"{d['risk']:.3f}"; wr = "n/a" if d["worst_region_risk"] is None else f"{d['worst_region_risk']:.3f}"
                ok = "n/a" if d["frac_test_regions_ok"] is None else f"{d['frac_test_regions_ok']:.2f}"
                log(f"  {tag:12s} {kind} coverage {d['coverage']:.2f} (exact {d['coverage_fine']:.2f} + group only {d['coverage_coarse']:.2f}) | risk {rk} | worst region {wr} | regions within alpha {ok}")
    save_json(res, f"{out}/checks_regions.json"); log(f"saved {out}/checks_regions.json")


# ------------------------------------------------------------------------------------------------ check 5: classes (data view)
def check_classes(a, dev, out, log):
    model, ck, tr, ca, te, info = load(a, dev, log); names = info["names"]; K = len(names); seg = train_seg(tr, info, dev); res = {}
    for split, D in [("train", tr), ("calibration", ca), ("test", te)]:
        rn = info.get("reg_names", {}).get(split, []); reg = D["reg"].numpy(); lab = D["lab"].numpy()
        log(f"{split.upper()}: label share per region (%), then how the frozen segmenter reads the TRUE colour (accuracy vs label)")
        log(f"  {'region':24s}" + "".join(f"{n:>9s}" for n in names) + f"{'seg acc':>9s}")
        for r in np.unique(reg):
            m = reg == r; l = lab[m]; share = np.bincount(l.ravel(), minlength=K) / l.size; ok, n = 0, 0
            with torch.no_grad():
                for i in np.where(m)[0][::max(1, int(m.sum()) // 40)]:
                    x = seg(D["L"][i:i + 1].to(dev), D["ab"][i:i + 1].to(dev)).argmax(1).cpu().numpy(); ok += (x == lab[i:i + 1]).sum(); n += x.size
            nm = rn[r] if r < len(rn) else f"{split}{r}"; res[f"{split}/{nm}"] = {"share": share.tolist(), "seg_acc_true_colour": float(ok / max(n, 1))}
            log(f"  {nm:24s}" + "".join(f"{100 * x:9.1f}" for x in share) + f"{ok / max(n, 1):9.2f}")
    save_json(res, f"{out}/checks_classes.json"); log(f"saved {out}/checks_classes.json")


# ------------------------------------------------------------------------------------------------ check 6: labelscore (selective risk against the TRUE labels)
def auroc(score, bad, n=2_000_000, seed=0):
    """probability that a randomly chosen WRONG pixel gets a higher score than a randomly chosen RIGHT pixel (0.5 = score carries no information)"""
    rng = np.random.default_rng(seed); i = rng.choice(len(score), min(n, len(score)), replace=False); s, b = score[i], bad[i] > 0.5
    if b.all() or (~b).all():
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(np.float64) + 1
    return float((r[b].sum() - b.sum() * (b.sum() + 1) / 2) / (b.sum() * (~b).sum()))


def check_labelscore(a, dev, out, log):
    """Which score tells us when the predicted class is WRONG (against WorldCover)?  Compares the ambiguity score used so far, the class head's own
    confidence, and a small logistic score FITTED on the calibration regions to predict a wrong class.  Risk = wrong class vs the label."""
    model, ck, tr, ca, te, info = load(a, dev, log); names = info["names"]; K = len(names)
    Pc, Pt = predict(model, ca, dev, ck["use_meta"], S=a.samples), predict(model, te, dev, ck["use_meta"], S=a.samples)
    ec = (Pc["pred"].flatten().numpy() != ca["lab"].numpy().ravel()).astype(np.float32); et = (Pt["pred"].flatten().numpy() != te["lab"].numpy().ravel()).astype(np.float32)
    flat = lambda P, k: P[k].flatten().numpy().astype(np.float32)
    def feats(P):                                                       # every feature is a per-pixel array at HR resolution
        f = {"spread": flat(P, "spread"), "entropy": flat(P, "ent"), "1-pmax": 1 - flat(P, "pmax"), "1-margin": 1 - flat(P, "margin"), "thermal": flat(P, "sr")}
        pr = P["pred"].flatten().numpy()
        for k in range(K): f[f"pred_{names[k]}"] = (pr == k).astype(np.float32)
        return f
    fc, ft = feats(Pc), feats(Pt); keys = list(fc); mu = {k: (float(fc[k].mean()), float(fc[k].std()) + 1e-8) for k in keys}
    rng = np.random.default_rng(0); sel = rng.choice(len(ec), min(400000, len(ec)), replace=False)
    X = torch.tensor(np.stack([(fc[k][sel] - mu[k][0]) / mu[k][1] for k in keys], 1)); y = torch.tensor(ec[sel])
    lin = torch.nn.Linear(len(keys), 1); opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)
    def closure():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(lin(X).squeeze(1), y) + 1e-4 * (lin.weight ** 2).sum(); l.backward(); return l
    opt.step(closure); w = lin.weight.detach().numpy().ravel(); b = float(lin.bias.detach())
    learned = lambda f: sum(w[i] * (f[k] - mu[k][0]) / mu[k][1] for i, k in enumerate(keys)) + b
    _, sc = abstention_report(Pc, (ec > 0).astype(float), Pt, et, te["lab"], names, 0.1, True, te, return_scores=True)
    scores = {"ambiguity (spread+entropy, as before)": sc["ambiguity"], "entropy (class head)": (fc["entropy"], ft["entropy"]), "1 - max probability (class head)": (fc["1-pmax"], ft["1-pmax"]),
              "spread (colour samples)": (fc["spread"], ft["spread"]), "LEARNED on label error": (learned(fc), learned(ft)), "random": (np.random.rand(len(ec)), np.random.rand(len(et)))}
    log("learned score weights (z-scored features): " + ", ".join(f"{k} {w[i]:+.2f}" for i, k in enumerate(keys)))
    log(f"full coverage: class-head label error calibration {ec.mean():.3f} | test {et.mean():.3f}")
    rn = info.get("reg_names", {}).get("test", []); npx = te["lab"].shape[-1] * te["lab"].shape[-2]; rt = np.repeat(te["reg"].numpy(), npx)
    log(f"{'score':40s} {'AUROC':>6s} {'err@50%':>8s} {'err@70%':>8s} {'err@90%':>8s} {'AURC':>6s}   (error of the committed pixels at that coverage, test regions; lower is better)")
    res = {}; per = {}
    for name, (s_c, s_t) in scores.items():
        g, r = M.risk_coverage(s_t, et); at = [float(r[int(c * 96) - 1]) for c in (0.5, 0.7, 0.9)]
        res[name] = {"auroc": auroc(s_t, et), "err_at_50": at[0], "err_at_70": at[1], "err_at_90": at[2], "aurc": float(r.mean())}
        log(f"{name:40s} {res[name]['auroc']:6.3f} {at[0]:8.3f} {at[1]:8.3f} {at[2]:8.3f} {r.mean():6.3f}")
        tau = float(np.quantile(s_c[::7], 0.7)); commit = s_t <= tau                       # threshold that commits 70% of the CALIBRATION pixels
        per[name] = {(rn[q] if q < len(rn) else f"test{q}"): (float(commit[rt == q].mean()), float(et[commit & (rt == q)].mean()) if (commit & (rt == q)).any() else None, float(et[rt == q].mean())) for q in np.unique(rt)}
    log("per test region, threshold fixed on calibration to commit 70% there:  coverage / error of committed / error with no abstention")
    for name in ["ambiguity (spread+entropy, as before)", "1 - max probability (class head)", "LEARNED on label error"]:
        log(f"  {name}: " + "  ".join(f"{k}: {v[0]:.2f}/{'n/a' if v[1] is None else format(v[1], '.3f')}/{v[2]:.3f}" for k, v in per[name].items()))
    res["per_region_at_70"] = per; res["weights"] = {k: float(w[i]) for i, k in enumerate(keys)}
    log("reading: AUROC 0.5 = the score says nothing about correctness; a useful score is clearly above 0.5 AND lowers the error of the committed pixels in EVERY region.")
    save_json(res, f"{out}/checks_labelscore.json"); log(f"saved {out}/checks_labelscore.json")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--check", required=True, choices=["graded", "consistency", "palette", "regions", "classes", "labelscore"])
    ap.add_argument("--ckpt", default="runs/v3_chroma/arc.pt"); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.05, 0.07, 0.10]); ap.add_argument("--label-alphas", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30], help="alphas for the label-error risk (its full-coverage value is much higher than class-flip)")
    ap.add_argument("--region-q", type=float, default=0.2)
    ap.add_argument("--score", default="ambiguity", choices=["ambiguity", "spread", "entropy"])
    ap.add_argument("--groups", default="water|forest,cropland|bare,built", help="coarse groups for graded commit: groups separated by |, classes by ,")
    a = ap.parse_args(); dev, out, log = setup(a, "v4_checks")
    {"graded": check_graded, "consistency": check_consistency, "palette": check_palette, "regions": check_regions, "classes": check_classes, "labelscore": check_labelscore}[a.check](a, dev, out, log)


if __name__ == "__main__":
    main()

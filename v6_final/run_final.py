"""FINAL RUN - one script for every method, so all of them share the split, the judge, the step budget and the evaluation code.
  --method arc      : ARC-IR. Stage 1 trains the structure branch (SR) for --stage1-steps; stage 2 freezes it and trains material head + flow (+ hybrid point head) for --steps.
  --method reg      : L1 U-Net colour regression              --method regvar : same U-Net, 4 outputs, Gaussian NLL (learned per-pixel variance)
  --method pix2pix  : the same U-Net as a conditional GAN     --method sr     : residual SR U-Net
Saves into --out: results.json, model.pt, and float16 predictions (cal_*.npy, test_*.npy) that v6_final/aggregate.py turns into the final tables.
"""
import argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common.utils import add_common_args, setup, get_data, to_dev, batch, cosine_lr, Timer, save_json
from common import metrics as M
from common.models import ARC, edges
from common.train import train_arc
from common.judge import load_judges
from common.tta import predict_arc, predict_unet
from v1_baselines.run_fair import UNet, Critic, make_input


def sync(): torch.cuda.synchronize() if torch.cuda.is_available() else None


def ms_per_tile(fn, D, n=32):
    sub = {k: v[:n] for k, v in D.items() if hasattr(v, "shape")}; fn(sub); sync(); t0 = time.time(); fn(sub); sync(); return (time.time() - t0) * 1000 / n


def colour_metrics(ab, D, seg, dev):
    ft = M.class_flips(seg, D, ab, dev)
    bad = 0
    with torch.no_grad():
        for i in range(0, D["L"].shape[0], 32):
            bad += (seg(D["L"][i:i + 32].to(dev), ab[i:i + 32].to(dev)).argmax(1).cpu() != D["lab"][i:i + 32]).sum().item()
    return {"chroma_mae": float((ab - D["ab"]).abs().mean()), "class_flip": float(ft.mean()), "label_error_seg_read": bad / D["lab"].numel()}


def save_preds(P, out, split, keys):
    for k in keys:
        if k in P: np.save(f"{out}/{split}_{k}.npy", P[k].half().numpy())


def train_structure(model, tr, a, dev, log):
    lr1 = a.lr if a.lr1 is None else a.lr1; we = a.w_edge if a.w_edge1 is None else a.w_edge1
    st = model.st; opt = torch.optim.AdamW(st.parameters(), lr=lr1, weight_decay=1e-4); tm = Timer(a.stage1_steps, log); N = len(tr["hr"]); st.train()
    for step in range(a.stage1_steps):
        cosine_lr(opt, lr1, step, a.stage1_steps); i = torch.randint(0, N, (a.bs,)); b = to_dev(batch(tr, i), dev)
        _, sr = st(b["lr"]); l1, le = F.l1_loss(sr, b["hr"]), F.l1_loss(edges(sr), edges(b["L"])); loss = l1 + we * le
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(st.parameters(), 1.0); opt.step(); tm.tick(step)
        if step % 500 == 0 or step == a.stage1_steps - 1: log(f"    [stage1] step {step:5d} l1={l1.item():.4f} edge={le.item():.3f}")
    st.eval()


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--method", default="arc", choices=["arc", "reg", "regvar", "pix2pix", "sr"]); ap.add_argument("--inputs", default="full", choices=["thermal", "full"])
    ap.add_argument("--width", type=int, default=48, help="U-Net width of the baselines (48 = 1.07 M parameters)")
    ap.add_argument("--width-mult", type=int, default=4, help="ARC-IR structure-branch width multiplier (4 = about 1.3 M parameters, 1 = the old 0.14 M)")
    ap.add_argument("--no-hybrid", action="store_true", help="ARC-IR without the deterministic point head (colour = mean of flow samples)")
    ap.add_argument("--stage1-steps", type=int, default=30000); ap.add_argument("--w-edge", type=float, default=0.05)
    ap.add_argument("--lr1", type=float, default=None, help="stage-1 (structure) learning rate; default = --lr"); ap.add_argument("--w-edge1", type=float, default=None, help="stage-1 edge weight; default = --w-edge")
    ap.add_argument("--task-w", type=float, default=0.0, help="ARC-IR task-driven variant: weight of the second judge's loss on the point colour (0 = off)")
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--tta-samples", type=int, default=4, help="flow samples PER view when 8-fold TTA is on (32 in total)")
    ap.add_argument("--l1", type=float, default=100.0); ap.add_argument("--judge-dir", default=None); ap.add_argument("--judge-steps", type=int, default=3000)
    ap.add_argument("--no-save-preds", action="store_true")
    a = ap.parse_args(); tag = a.method + ("_task" if a.task_w > 0 else "") + ("_flowmean" if a.no_hybrid else "")
    dev, out, log = setup(a, f"final_{tag}_s{a.seed}"); tr, ca, te, info = get_data(a, log); seg, seg2 = load_judges(a.judge_dir, tr, info, dev, a.judge_steps)
    log(f"golden rules: split {a.regions} | judge {a.judge_dir or 'trained inline (test only)'} | seed {a.seed} | steps {a.steps}" + (f" (+ {a.stage1_steps} structure steps)" if a.method == "arc" else ""))
    res = {"method": tag, "seed": a.seed, "steps": a.steps, "inputs": a.inputs}; hr = tr["hr"].shape[-2:]; keys = ["ab", "spread", "ent", "pred", "ab_flow", "pmax"]
    if a.method == "arc":
        model = ARC("arc", info, a.structure, a.scale, a.weights, aux_ch=info["aux_ch"], width_mult=a.width_mult, hybrid=not a.no_hybrid).to(dev)
        res["params"] = sum(p.numel() for p in model.parameters()); log(f"ARC-IR: {res['params'] / 1e6:.3f} M parameters (structure branch {sum(p.numel() for p in model.st.parameters()) / 1e6:.3f} M)")
        train_structure(model, tr, a, dev, log)
        train_arc(model, tr, a, dev, log, w_inv=0.0, freeze_st=True, w_edge=a.w_edge, task_judge=seg2 if a.task_w > 0 else None, task_w=a.task_w)
        f0 = lambda D: predict_arc(model, D, dev, False, a.samples); f1 = lambda D: predict_arc(model, D, dev, True, a.tta_samples)
    else:
        cin = make_input({k: tr[k][:2] for k in ["lr", "m"] + (["aux"] if "aux" in tr else [])}, hr, a.inputs).shape[1]
        cout = {"sr": 1, "regvar": 4}.get(a.method, 2); net = UNet(cin, cout, a.width).to(dev); params = list(net.parameters()); res["params"] = sum(p.numel() for p in params)
        log(f"{a.method}: {res['params'] / 1e6:.3f} M parameters ({cin} input channels)")
        opt = torch.optim.AdamW(params, lr=a.lr / 4); tm = Timer(a.steps, log); N = len(tr["hr"])
        if a.method == "pix2pix": D = Critic(3, a.width).to(dev); optd = torch.optim.AdamW(D.parameters(), lr=a.lr / 4, betas=(0.5, 0.999))
        for step in range(a.steps):
            cosine_lr(opt, a.lr / 4, step, a.steps); i = torch.randint(0, N, (a.bs,)); b = to_dev(batch(tr, i), dev); x = make_input(b, hr, a.inputs); y = net(x)
            if a.method == "sr": loss = F.l1_loss(x[:, :1] + y, b["hr"])
            elif a.method == "reg": loss = F.l1_loss(y, b["ab"])
            elif a.method == "regvar":
                lv = y[:, 2:].clamp(-10, 4); loss = (0.5 * (lv + (y[:, :2] - b["ab"]) ** 2 * torch.exp(-lv))).mean()
            else:
                real, fake = torch.cat([b["hr"], b["ab"]], 1), torch.cat([b["hr"], y.detach()], 1)
                ld = 0.5 * (F.mse_loss(D(real), torch.ones_like(D(real))) + F.mse_loss(D(fake), torch.zeros_like(D(fake)))); optd.zero_grad(); ld.backward(); optd.step()
                loss = F.mse_loss(D(torch.cat([b["hr"], y], 1)), torch.ones_like(D(real))) + a.l1 * F.l1_loss(y, b["ab"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tm.tick(step)
            if step % 500 == 0 or step == a.steps - 1: log(f"    step {step:5d} loss={loss.item():.4f}")
        f0 = lambda D: predict_unet(net, D, dev, make_input, a.method, a.inputs, False); f1 = lambda D: predict_unet(net, D, dev, make_input, a.method, a.inputs, True)
    res["ms_per_tile"] = {"no_tta": ms_per_tile(f0, te), "tta8": ms_per_tile(f1, te)}; log(f"inference ms/tile: {res['ms_per_tile']}")
    for name, f in (("no_tta", f0), ("tta8", f1)):
        P = f(te); r = {}
        if a.method == "sr" or a.method == "arc":
            r["sr_psnr"], r["sr_ssim"] = M.psnr(P["sr"], te["hr"]), M.ssim(P["sr"], te["hr"])
        if a.method != "sr":
            r.update(colour_metrics(P["ab"], te, seg, dev))
            if a.method == "arc":
                r["cls_acc"] = float((P["pred"] == te["lab"]).float().mean()); r["label_error_head"] = 1 - r["cls_acc"]
                if not a.no_hybrid: r["flow_mean"] = colour_metrics(P["ab_flow"], te, seg, dev)           # ablation: colour = mean of the flow samples
        res[name] = r; log(f"test [{name}]: " + " | ".join(f"{k} {v:.4f}" for k, v in r.items() if not isinstance(v, dict)))
        if r.get("flow_mean"): log(f"      flow-mean ablation: " + " | ".join(f"{k} {v:.4f}" for k, v in r["flow_mean"].items()))
        if name == "tta8" and not a.no_save_preds and a.method != "sr":
            save_preds(P, out, "test", keys); save_preds(f(ca), out, "cal", keys); log("saved predictions (float16)")
    if a.method == "arc" or a.method == "sr": torch.save({"state": (model if a.method == "arc" else net).state_dict(), "args": vars(a)}, f"{out}/model.pt")
    else: torch.save({"state": net.state_dict(), "args": vars(a)}, f"{out}/model.pt")
    save_json(res, f"{out}/results.json"); log(f"saved {out}/results.json")


if __name__ == "__main__":
    main()

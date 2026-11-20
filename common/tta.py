"""8-fold test-time augmentation (4 rotations x 2 flips = the dihedral group) and the shared prediction functions for every method."""
import numpy as np, torch, torch.nn.functional as F
from .models import sample_chroma


def d4(x, k):                                   # k = 0..7 : flip (k >= 4) then rotate by (k % 4) * 90 deg
    if k >= 4: x = x.flip(-1)
    return torch.rot90(x, k % 4, (-2, -1))


def d4inv(x, k):
    x = torch.rot90(x, -(k % 4), (-2, -1))
    return x.flip(-1) if k >= 4 else x


@torch.no_grad()
def predict_arc(model, D, dev, tta=False, S=8, steps=8, bs=4):
    """returns per-tile tensors: sr, ab (point estimate: hybrid head if present, else flow mean), ab_flow (flow mean), spread (std of flow samples,
    pooled over all TTA views), ent / pmax / pred of the class head (probabilities averaged over views)"""
    model.eval(); ks = range(8) if tta else [0]; out = {k: [] for k in ["sr", "ab", "ab_flow", "spread", "ent", "pmax", "pred"]}
    for i in range(0, len(D["hr"]), bs):
        lr, m = D["lr"][i:i + bs].to(dev), D["m"][i:i + bs].to(dev); aux = D["aux"][i:i + bs].to(dev) if "aux" in D else None
        sr = pt = pr = 0; samples = []
        for k in ks:
            o = model.encode(d4(lr, k), m, d4(aux, k) if aux is not None else None)
            sr = sr + d4inv(o["sr"], k); pr = pr + d4inv(F.softmax(o["logits"], 1), k)
            if getattr(model, "head", None) is not None: pt = pt + d4inv(model.head(o["cond"]), k)
            x = sample_chroma(model, o, S, steps)                                    # B,S,2,H,W
            samples.append(torch.stack([d4inv(x[:, j], k) for j in range(x.shape[1])], 1))
        n = len(list(ks)); smp = torch.cat(samples, 1); flow = smp.mean(1); pr = pr / n
        out["sr"].append((sr / n).cpu()); out["ab_flow"].append(flow.cpu()); out["ab"].append((pt / n if getattr(model, "head", None) is not None else flow).cpu())
        out["spread"].append(smp.std(1, unbiased=False).mean(1).cpu())
        out["ent"].append((-(pr * torch.log(pr + 1e-8)).sum(1)).cpu()); t2 = pr.topk(2, 1).values; out["pmax"].append(t2[:, 0].cpu()); out["pred"].append(pr.argmax(1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def predict_unet(net, D, dev, make_input, method, inputs, tta=False, bs=8):
    """method: reg | sr | pix2pix | regvar (regvar returns mean + a learned per-pixel sigma). Output dict: ab or sr, spread (sigma, regvar only)"""
    net.eval(); ks = range(8) if tta else [0]; res = {"ab": [], "sr": [], "spread": []}; hr = D["hr"].shape[-2:]
    with torch.no_grad():
        for i in range(0, len(D["hr"]), bs):
            lr, m = D["lr"][i:i + bs].to(dev), D["m"][i:i + bs].to(dev); aux = D["aux"][i:i + bs].to(dev) if "aux" in D else None
            acc = var = 0
            for k in ks:
                b = {"lr": d4(lr, k), "m": m}
                if aux is not None: b["aux"] = d4(aux, k)
                x = make_input(b, hr, inputs); y = net(x)
                if method == "sr": y = x[:, :1] + y
                if method == "regvar":
                    mu, lv = y[:, :2], y[:, 2:].clamp(-10, 4); acc = acc + d4inv(mu, k); var = var + d4inv(torch.exp(lv), k)
                else:
                    acc = acc + d4inv(y, k)
            n = len(list(ks)); acc = acc / n
            if method == "sr": res["sr"].append(acc.cpu())
            else: res["ab"].append(acc.cpu())
            if method == "regvar": res["spread"].append(torch.sqrt(var / n).mean(1).cpu())
    return {k: torch.cat(v) for k, v in res.items() if v}

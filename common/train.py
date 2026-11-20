import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .models import edges
from .utils import batch, to_dev, cosine_lr, Timer


def train_arc(model, D, args, dev, log, w_inv=0.0, use_meta=True, freeze_st=False, steps=None,
              w_sr=1.0, w_cls=0.5, w_edge=0.3, task_judge=None, task_w=0.0):
    steps = steps or args.steps
    if freeze_st:
        for p in model.st.parameters():
            p.requires_grad_(False)
        model.st.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    N, tm = len(D["hr"]), Timer(steps, log)
    model.train()
    if freeze_st:
        model.st.eval()
    for step in range(steps):
        cosine_lr(opt, args.lr, step, steps)
        i = torch.randint(0, N, (args.bs,))
        b = to_dev(batch(D, i), dev)
        m = b["m"] * (1.0 if use_meta else 0.0)
        o = model.encode(b["lr"], m, b.get("aux"))
        l_sr = F.l1_loss(o["sr"], b["hr"]); l_edge = F.l1_loss(edges(o["sr"]), edges(b["L"]))
        if "soft" in b:                              # soft land-cover fractions of each 30 m pixel (mixed pixels are common)
            l_cls = -((b["soft"].float() / 100.0) * F.log_softmax(o["logits"], 1)).sum(1).mean()
        else:
            l_cls = F.cross_entropy(o["logits"], b["lab"])
        if model.mode == "reg":
            l_col = F.l1_loss(model.head(o["cond"]), b["ab"])
        else:
            x1, x0 = b["ab"], torch.randn_like(b["ab"])
            t = torch.rand(x1.shape[0], device=dev)
            xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
            l_col = F.mse_loss(model.flow(xt, t, o["cond"]), x1 - x0)
        loss = w_cls * l_cls + l_col
        if model.mode != "reg" and getattr(model, "hybrid", False):          # point-estimate head, trained beside the flow
            ab_pt = model.head(o["cond"]); l_pt = F.l1_loss(ab_pt, b["ab"]); loss = loss + l_pt
            if task_judge is not None and task_w > 0:                         # optional: a SECOND judge (never the evaluation judge) reads our colour
                l_task = F.cross_entropy(task_judge(b["L"], ab_pt), b["lab"]); loss = loss + task_w * l_task
        if not freeze_st:
            loss = loss + w_sr * l_sr + w_edge * l_edge
        l_inv = torch.zeros((), device=dev)
        if w_inv > 0 and "zm" in o:
            j = D["pair"][i]; v = j >= 0
            if v.any():
                b2 = to_dev(batch(D, j[v]), dev)
                o2 = model.encode(b2["lr"], b2["m"] * (1.0 if use_meta else 0.0), b2.get("aux"))
                vv = v.to(dev)
                l_inv = F.mse_loss(o["zm"][vv], o2["zm"]) + F.mse_loss(F.softmax(o["logits"], 1)[vv], F.softmax(o2["logits"], 1))
                loss = loss + w_inv * l_inv
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        tm.tick(step)
        if step % 100 == 0 or step == steps - 1:
            log(f"    step {step:5d} sr={l_sr.item():.4f} cls={l_cls.item():.3f} edge={l_edge.item():.3f} "
                f"col={l_col.item():.3f} inv={float(l_inv.detach()):.4f}")
    return model.eval()

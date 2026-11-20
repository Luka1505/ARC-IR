import numpy as np, torch, torch.nn.functional as F


def psnr(a, b, rng=1.0):
    return float(10 * torch.log10(rng ** 2 / F.mse_loss(a, b)))


def _gauss(n=11, s=1.5, dev="cpu"):
    x = torch.arange(n, dtype=torch.float32, device=dev) - n // 2
    g = torch.exp(-x ** 2 / (2 * s * s)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, rng=1.0):
    w = _gauss(dev=a.device); C1, C2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    f = lambda x: F.conv2d(x, w)
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    return float((((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).mean())


_LP = None
def lpips_rgbish(a, b):
    """LPIPS needs 3-channel input; the thermal channel is repeated. Returns None if `lpips` is not installed."""
    global _LP
    try:
        import lpips
    except ImportError:
        return None
    if _LP is None:
        _LP = lpips.LPIPS(net="alex").to(a.device)
    r = lambda x: (x.repeat(1, 3, 1, 1) * 2 - 1)
    with torch.no_grad():
        return float(_LP(r(a), r(b)).mean())


def risk_coverage(score, flip, grid=np.linspace(0.05, 1.0, 96)):
    o = np.argsort(score, kind="stable")
    cum = np.cumsum(flip[o].astype(np.float64)) / np.arange(1, len(o) + 1)
    return grid, cum[(grid * len(o)).astype(int).clip(1) - 1]


def calibrate(score_cal, flip_cal, alpha):
    """largest commit set whose class-flip rate on the calibration regions is <= alpha (returns the score threshold)"""
    o = np.argsort(score_cal, kind="stable")
    cum = np.cumsum(flip_cal[o].astype(np.float64)) / np.arange(1, len(o) + 1)
    ok = np.where(cum <= alpha)[0]
    return score_cal[o][ok[-1]] if len(ok) else -np.inf


def zfit(x, ref):
    return (x - ref.mean()) / (ref.std() + 1e-8)


def fit_ambiguity(spread_c, ent_c, flip_c, n=300000, seed=0):
    """logistic regression of class flip on [spread, entropy], fitted on the calibration regions"""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(flip_c), min(n, len(flip_c)), replace=False)
    X = torch.tensor(np.stack([zfit(spread_c[sel], spread_c), zfit(ent_c[sel], ent_c)], 1), dtype=torch.float32)
    y = torch.tensor(flip_c[sel], dtype=torch.float32)
    lin = torch.nn.Linear(2, 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=100)
    def closure():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(lin(X).squeeze(1), y); l.backward(); return l
    opt.step(closure)
    return lin.weight.detach().numpy().ravel(), float(lin.bias.detach())


@torch.no_grad()
def class_flips(seg, D, ab_pred, dev, bs=32):
    """1 where the frozen segmenter's class for (L, predicted colour) differs from its class for (L, true colour)"""
    out = []
    for i in range(0, D["L"].shape[0], bs):
        L, ab, p = D["L"][i:i + bs].to(dev), D["ab"][i:i + bs].to(dev), ab_pred[i:i + bs].to(dev)
        out.append((seg(L, p).argmax(1) != seg(L, ab).argmax(1)).float().flatten().cpu())
    return torch.cat(out).numpy()

"""Synthetic 5-class world used to test every stage end to end (no real data needed).
Same tensor layout as the real Landsat loader, so switching data sources changes nothing else."""
import numpy as np, torch
from scipy.ndimage import gaussian_filter

NAMES = ["water", "forest", "cropland", "bare", "built"]
K = len(NAMES)
TH = np.array([[0.20, 0.30, 0.01], [0.30, 0.50, 0.06], [0.55, 0.90, 0.03],
               [0.58, 0.95, 0.03], [0.75, 0.60, 0.05]], np.float32)   # base, gain vs thermal state, texture
LUM = np.array([0.20, 0.35, 0.55, 0.70, 0.60], np.float32)
AB = np.array([[-0.2, -0.8], [-0.7, 0.4], [-0.5, 0.5], [0.4, 0.5], [0.1, -0.1]], np.float32)
INFO = {"K": K, "names": NAMES, "mdim": 4}


def _noise(rng, sigma, n):
    x = gaussian_filter(rng.normal(size=(n, n)), sigma)
    return (x / x.std()).astype(np.float32)


def _tile(rng, bias, dT, n):
    f = np.stack([gaussian_filter(rng.normal(size=(n, n)), 5) for _ in range(K)])
    f = f / f.std(axis=(1, 2), keepdims=True) + bias[:, None, None]
    lab = f.argmax(0)
    elev = rng.uniform(0, 1)
    hr_l, L_l, ab_l, m_l = [], [], [], []
    for _ in range(2):                                   # two dates over the same ground
        doy, sun = rng.uniform(0, 1), rng.uniform(0.3, 1.0)
        s = 0.5 * sun + 0.3 * np.cos(2 * np.pi * doy) - 0.3 * elev + 0.2
        dry = float(np.clip(0.5 + 0.5 * np.sin(2 * np.pi * doy), 0, 1))
        T = TH[lab, 0] + TH[lab, 1] * s * 0.5 + dT + TH[lab, 2] * _noise(rng, 0.8, n) + 0.01 * rng.normal(size=(n, n))
        L = LUM[lab] + 0.04 * _noise(rng, 1.5, n)
        ab = AB[lab].copy()
        ab[..., 0] += (lab == 2) * 0.9 * dry
        ab[..., 1] -= (lab == 1) * 0.2 * dry
        ab = ab.transpose(2, 0, 1) + 0.02 * rng.normal(size=(2, n, n))
        hr_l.append(T[None]); L_l.append(L[None]); ab_l.append(ab)
        m_l.append([np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy), sun, elev])
    return np.stack(hr_l), np.stack(L_l), np.stack(ab_l), np.array(m_l), lab


def make_region(r, n_tiles, n, scale, seed=0):
    rng = np.random.default_rng(seed + 1000 * r)
    bias, dT = rng.normal(0, 0.6, K), rng.normal(0, 0.04)
    tiles = [_tile(rng, bias, dT, n) for _ in range(n_tiles)]
    cols = list(zip(*tiles))
    out = {}
    for name, c in zip(["hr", "L", "ab", "m"], cols[:4]):
        a = np.stack(c).astype(np.float32)
        out[name] = torch.from_numpy(a.reshape(-1, *a.shape[2:]))        # tile-major [t0d0, t0d1, t1d0, ...]
    out["lab"] = torch.from_numpy(np.repeat(np.stack(cols[4]), 2, axis=0)).long()
    return out


def degrade(hr, scale):
    """thermal LR from HR: blur + subsample (stand-in for the 100 m sensor), plus sensor noise"""
    import torch.nn.functional as F
    lr = F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", antialias=True, align_corners=False)
    return lr + 0.01 * torch.randn_like(lr)


def build(tiles=120, ncal=80, ntest=70, n=64, scale=2, seed=0):
    cat = lambda rs: {k: torch.cat([r[k] for r in rs]) for k in rs[0]}
    train = cat([make_region(r, tiles, n, scale, seed) for r in range(4)])      # regions 0-3
    cal = make_region(4, ncal, n, scale, seed)                                   # region 4: calibration
    test = cat([make_region(r, ntest, n, scale, seed) for r in (5, 6)])          # regions 5-6: held-out geography
    for D in (train, cal, test):
        g = torch.Generator().manual_seed(seed + 7)
        D["lr"] = degrade(D["hr"], scale)
        D["pair"] = torch.arange(D["hr"].shape[0]) ^ 1                           # other date of the same tile
    return train, cal, test, INFO

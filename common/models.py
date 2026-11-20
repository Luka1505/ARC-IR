import torch, torch.nn as nn, torch.nn.functional as F


def conv(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.GELU())


class Structure(nn.Module):
    """Small SR network (stand-in). Replace with the pretrained backbone via build_structure('hat', ...)."""
    def __init__(s, c=32, scale=2):
        super().__init__()
        s.scale, s.feat_dim = scale, c
        s.enc = nn.Sequential(conv(1, c), conv(c, c), conv(c, c))
        s.up = nn.Sequential(nn.Conv2d(c, c * scale * scale, 3, padding=1), nn.PixelShuffle(scale), nn.GELU())
        s.dec = nn.Sequential(conv(c, c), conv(c, c))
        s.sr = nn.Conv2d(c, 1, 3, padding=1)

    def forward(s, x):
        f = s.dec(s.up(s.enc(x)))
        return f, s.sr(f) + F.interpolate(x, scale_factor=s.scale, mode="bilinear", align_corners=False)


class HATAdapter(nn.Module):
    """Pretrained HAT (basicsr) for 1-channel thermal: repeat to 3 channels, average the output.
    Architecture tested offline with random weights (v2_structure/test_hat.py); the pretrained weights are not tested here. Match the hyper-parameters to your checkpoint
    (HAT-L: depths=(6,)*12, num_heads=(6,)*12, embed_dim=180)."""
    feat_dim = 1

    def __init__(s, weights, scale=2, large=True):
        super().__init__()
        from .hat_arch import HAT                      # vendored copy of the official HAT (no basicsr needed); needs `pip install einops`
        n = 12 if large else 6
        s.scale = scale
        s.net = HAT(img_size=64, patch_size=1, in_chans=3, embed_dim=180, depths=(6,) * n, num_heads=(6,) * n,
                    window_size=16, compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                    mlp_ratio=2., upscale=scale, img_range=1., upsampler="pixelshuffle", resi_connection="1conv")
        s.net.grad_ckpt = __import__("os").environ.get("ARC_CKPT", "1") == "1"                         # recompute layers in backward: HAT-L needs ~4 GB per tile otherwise
        if weights:                                    # weights=None builds the architecture with random weights (used by the offline test)
            sd = torch.load(weights, map_location="cpu")
            r = s.net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=False)
            if r.missing_keys:
                raise RuntimeError(f"HAT weights do not match the architecture ({len(r.missing_keys)} missing keys, e.g. {r.missing_keys[:3]}): check --structure hat (HAT-L) vs hat-s")
            print(f"HAT weights loaded from {weights} ({len(r.unexpected_keys)} unused keys)")

    def forward(s, x):
        y = s.net(x.repeat(1, 3, 1, 1)).mean(1, keepdim=True)
        return y, y


def build_structure(name, scale=2, weights=None, width_mult=1):
    if name == "small":
        return Structure(32 * width_mult, scale)
    if name in ("hat", "hat-s"):
        return HATAdapter(weights, scale, large=(name == "hat"))
    raise ValueError(name)


class Flow(nn.Module):
    def __init__(s, cin, c=32):
        super().__init__()
        s.net = nn.Sequential(conv(3 + cin, c), conv(c, c), conv(c, c), nn.Conv2d(c, 2, 3, padding=1))

    def forward(s, x, t, cond):
        tt = t[:, None, None, None].expand(-1, 1, x.shape[2], x.shape[3])
        return s.net(torch.cat([x, tt, cond], 1))


class ARC(nn.Module):
    """mode 'arc'    : chroma sees material code + thermal-state code + SR image (factorised)
       mode 'unfact' : same size, no factorisation
       mode 'reg'    : deterministic single-estimate colorizer (Pix2Pix-like), no sampling"""
    def __init__(s, mode, info, structure="small", scale=2, weights=None, zm=16, zt=8, c=32, aux_ch=0, width_mult=1, hybrid=False):
        super().__init__()
        s.mode, s.K, s.hybrid = mode, info["K"], hybrid
        s.st = build_structure(structure, scale, weights, width_mult)
        c = s.st.feat_dim                            # small: 32 * width_mult (width_mult=1 gives the old 32)
        s.aux_ch = aux_ch
        if aux_ch:                                   # extra inputs (other dates, terrain) enter beside the structure features
            s.aux = nn.Sequential(conv(aux_ch, 16), conv(16, 16)); c = c + 16
        s.mlp = nn.Sequential(nn.Linear(info["mdim"], 32), nn.GELU(), nn.Linear(32, zt))
        if mode == "arc":
            s.mat = nn.Sequential(conv(c + zt, 32), conv(32, zm))
            s.cls = nn.Conv2d(zm, s.K, 1)
            cc = zm + zt + 1
        else:
            s.cls = nn.Sequential(conv(c + zt, 32), nn.Conv2d(32, s.K, 1))
            cc = c + zt
        if mode == "reg" or hybrid:                  # hybrid: deterministic point colour from the same conditioning; the flow is kept for spread only
            s.head = nn.Sequential(conv(cc, 32), conv(32, 32), nn.Conv2d(32, 2, 3, padding=1))
        if mode != "reg":
            s.flow = Flow(cc, 32)

    def encode(s, lr, m, aux=None):
        f, sr = s.st(lr)
        if s.aux_ch:
            a = s.aux(aux.float()) if aux is not None else torch.zeros(f.shape[0], 16, *lr.shape[-2:], device=f.device)
            f = torch.cat([f, F.interpolate(a, size=f.shape[-2:], mode="bilinear", align_corners=False)], 1)
        zt = s.mlp(m)[:, :, None, None].expand(-1, -1, f.shape[2], f.shape[3])
        o = {"sr": sr, "f": f[:, :f.shape[1] - (16 if s.aux_ch else 0)]}
        if s.mode == "arc":
            zm = s.mat(torch.cat([f, zt], 1))
            o.update(zm=zm, logits=s.cls(zm), cond=torch.cat([zm, zt, sr], 1))
        else:
            o.update(logits=s.cls(torch.cat([f, zt], 1)), cond=torch.cat([f, zt], 1))
        return o


class Seg(nn.Module):
    """Frozen evaluation segmenter (L,a,b) -> class. On real data train it on WorldCover and do not use it in training."""
    def __init__(s, K):
        super().__init__()
        s.n = nn.Sequential(conv(3, 24), conv(24, 24), nn.Conv2d(24, K, 1))

    def forward(s, L, ab):
        return s.n(torch.cat([L, ab], 1))


_KX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def edges(x):
    kx = _KX.to(x)
    gx, gy = F.conv2d(x, kx, padding=1), F.conv2d(x, kx.transpose(2, 3), padding=1)
    e = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return e / (e.mean((1, 2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def sample_chroma(model, o, S=8, steps=8):
    """draw S chroma samples per tile by Euler-integrating the learned flow from noise"""
    B, _, H, W = o["cond"].shape
    c = o["cond"].repeat_interleave(S, 0)
    x = torch.randn(B * S, 2, H, W, device=c.device)
    for k in range(steps):
        x = x + model.flow(x, torch.full((B * S,), k / steps, device=c.device), c) / steps
    return x.view(B, S, 2, H, W)

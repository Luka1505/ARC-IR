"""Offline test of the enrich step + multi-date / terrain / soft-label loader + region-aware calibration. No network.
    python -m common.test_enrich"""
import os, sys, tempfile, numpy as np, yaml, torch
from data_tools.make_tiles import cut_tiles
from data_tools.enrich import soft_labels, slope_deg, patch_tiles
from common import landsat
from common.evalx import region_report

rng = np.random.default_rng(0); d = tempfile.mkdtemp(); H = W = 256
# 1) soft labels from a fake 10 m map: a 3x3 block that is 6 x class 40 (cropland) + 3 x class 10 (forest) -> soft 67/33, mode cropland
wc10 = np.full((6, 6), 40, np.uint8); wc10[:3, :3] = 10; wc10[0, 0] = 40
mode, soft = soft_labels(wc10)
assert soft.shape == (5, 2, 2)
assert int(mode[1, 1]) == 2 and soft[:, 0, 0].sum() in (99, 100, 101), (mode, soft[:, 0, 0])
assert abs(float(slope_deg(np.add.outer(np.arange(8) * 30.0, np.zeros(8))).mean()) - 45) < 1.0
# 2) fake region folders with several scenes at the same grid positions, then patch them
for r in ["r1", "r2", "r3", "r4", "r5", "r6"]:
    rd = os.path.join(d, r)
    for k, sc in enumerate(["s2021a", "s2021b", "s2022a"]):
        base = 300 + 4 * (hash(r) % 3)
        th = rng.random((H, W)) * 20 + 290; rgb = rng.random((3, H, W)) * 0.3; wc = rng.integers(0, 5, (H, W)).astype(np.uint8)
        cut_tiles(th, rgb, wc, np.ones((H, W), bool), [np.sin(k), np.cos(k), 0.5, 0.3], r, sc, rd, 128, 0.9)
    m, s_ = soft_labels(rng.choice([10, 40, 50, 80], (H * 3, W * 3)).astype(np.uint8)); dem = np.zeros((H, W), np.float16); sl = dem.copy()
    n, sk = patch_tiles(rd, m, s_, (rng.random((H, W)) * 800).astype(np.float16), (rng.random((H, W)) * 20).astype(np.float16))
    assert n == 12 and sk == 0, (n, sk)
    assert patch_tiles(rd, m, s_, dem * 0, sl * 0)[0] == 0            # idempotent
yaml.safe_dump({"train": ["r1", "r2", "r3"], "calibration": ["r4", "r5"], "test": ["r6"]}, open(os.path.join(d, "reg.yaml"), "w"))
# 3) loader with every new option
tr, ca, te, info = landsat.build(d, os.path.join(d, "reg.yaml"), scale=2, dates=3, terrain=True, soft=True, log=lambda *a: None)
assert info["aux_ch"] == 2 + 4 * 2 and tr["aux"].shape[1] == 10 and tr["aux"].shape[-1] == 64, tr["aux"].shape
assert tr["soft"].shape[1] == 5 and tr["reg"].shape[0] == tr["hr"].shape[0] and info["reg_names"]["test"] == ["r6"]
assert (tr["aux"][:, -1].float() == 1).all()                         # every tile has 2 other dates here (valid flags = 1)
# 4) the model runs with aux and soft labels
from common.models import ARC
from common.train import train_arc
class A: steps = 4; bs = 4; lr = 1e-3
for mode_ in ["arc", "reg", "unfact"]:
    mdl = ARC(mode_, info, "small", 2, None, aux_ch=info["aux_ch"])
    train_arc(mdl, tr, A, torch.device("cpu"), lambda *a: None, w_inv=1.0 if mode_ == "arc" else 0.0)
# 5) region-aware calibration on synthetic scores: region 'hard' is much harder than the rest
n = 20000; reg_c = np.repeat(np.arange(4), n // 4); s_c = rng.random(n); fc = (rng.random(n) < 0.02 + 0.3 * s_c * (1 + 2 * (reg_c == 3))).astype(float)
reg_t = np.repeat(np.arange(2), n // 2); s_t = rng.random(n); ft = (rng.random(n) < 0.02 + 0.3 * s_t * (1 + 2 * (reg_t == 1))).astype(float)
rr = region_report(s_c, s_t, fc, ft, reg_c, reg_t, np.zeros(n, int), np.zeros(n, int), 0.05, 0.2, ["a"])
assert rr["region_quantile"]["coverage"] <= rr["pooled"]["coverage"] + 1e-9 and rr["region_quantile"]["worst_region_risk"] <= rr["pooled"]["worst_region_risk"] + 1e-9
print("enrich + multi-date + terrain + soft labels + region-aware calibration: OK (offline)")

"""Offline test: fake scene -> cut_tiles -> npz -> landsat.build -> a batch the models accept. Run: python -m common.test_data_tools"""
import os, tempfile, numpy as np, yaml, torch
from data_tools.make_tiles import cut_tiles, map_worldcover, rgb_to_lab, lab_to_rgb, pick_spread, bbox_to_bounds
from common import landsat
from common.models import ARC


def main():
    rng = np.random.default_rng(0); H = W = 512; d = tempfile.mkdtemp()
    parts = {"train": ["a"], "calibration": ["b"], "test": ["c"]}
    for part, regs in parts.items():
        for r in regs:
            for sc in ("s1", "s2"):                                      # two scenes of the same place -> date pairs
                wc = rng.choice([10, 30, 40, 50, 60, 80], size=(H, W)); wc = np.kron(wc[::64, ::64], np.ones((64, 64), int))
                n = cut_tiles(280 + 30 * rng.random((H, W)), 0.05 + 0.2 * rng.random((3, H, W)), map_worldcover(wc),
                              np.ones((H, W), bool), [0.1, 0.9, 0.6, 0.2], r, sc, os.path.join(d, r), size=256)
                assert n == 4, n
    yaml.safe_dump(parts, open(os.path.join(d, "regions.yaml"), "w"))
    tr, ca, te, info = landsat.build(d, os.path.join(d, "regions.yaml"), scale=2)
    assert (tr["pair"] >= 0).all(), "tiles of the same place should be paired across scenes"
    assert tr["hr"].shape == (8, 1, 256, 256) and tr["lr"].shape == (8, 1, 128, 128) and tr["ab"].shape == (8, 2, 256, 256)
    m = ARC("arc", info); o = m.encode(tr["lr"][:2], tr["m"][:2]); assert o["sr"].shape == tr["hr"][:2].shape
    L, ab = rgb_to_lab(np.ones((3, 4, 4))); assert abs(L.mean() - 1) < 0.01 and abs(ab).max() < 0.05      # white -> L=100, a=b=0
    rgb = rng.random((3, 8, 8)); L2, ab2 = rgb_to_lab(rgb); back = lab_to_rgb(L2, ab2)
    assert np.abs(back - np.moveaxis(rgb, 0, -1)).max() < 0.01, "lab_to_rgb should invert rgb_to_lab"
    class It:  # fake STAC item
        def __init__(s, d): s.properties = {"datetime": d}
    its = [It(f"2022-{m:02d}-15") for m in range(1, 13)] + [It("2021-03-01")]
    pick = pick_spread(its, 5); assert len(pick) == 5 and pick[0].properties["datetime"] == "2021-03-01" and pick[-1].properties["datetime"] == "2022-12-15"
    b = bbox_to_bounds([80.6, 15.7, 81.1, 16.2], 32644); assert b[0] % 30 == 0 and 40000 < b[2] - b[0] < 70000, b
    print("real-data format + tiling + scene selection + bbox conversion: OK (offline, fake scenes)")


if __name__ == "__main__":
    main()

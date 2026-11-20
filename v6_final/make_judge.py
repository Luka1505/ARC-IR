"""Train the primary judge (seed 123) and the independent second judge (seed 777) ONCE and save them. Every final run loads these files.
Run:  python -m v6_final.make_judge --data landsat --tiles-dir dataset_big --regions data_tools/regions_big.yaml --terrain --dates 3 --abs-temp --soft --color-target fixed --out runs/final_judge
"""
import argparse, time, torch
from common.utils import add_common_args, setup, get_data, train_seg
from v4_abstain.judge_check import seg_err


def main():
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--judge-steps", type=int, default=3000); a = ap.parse_args()
    dev, out, log = setup(a, "final_judge"); tr, ca, te, info = get_data(a, log)
    for name, seed in (("judge.pt", 123), ("judge2.pt", 777)):
        t0 = time.time(); s = train_seg(tr, info, dev, steps=a.judge_steps, seed=seed); torch.save(s.state_dict(), f"{out}/{name}")
        log(f"{name}: seed {seed}, {a.judge_steps} steps ({time.time() - t0:.0f}s) | label error on TRUE colour: calibration {seg_err(s, ca, dev):.3f} test {seg_err(s, te, dev):.3f}")


if __name__ == "__main__":
    main()

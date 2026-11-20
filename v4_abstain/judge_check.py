import argparse, time, torch
from common.utils import add_common_args, setup, get_data, train_seg

def seg_err(seg, D, dev, bs=64):
    wrong = tot = 0
    for i in range(0, D["L"].shape[0], bs):
        p = seg(D["L"][i:i + bs].to(dev), D["ab"][i:i + bs].to(dev)).argmax(1).cpu()
        wrong += (p != D["lab"][i:i + bs]).sum().item(); tot += p.numel()
    return wrong / tot

def main():
    ap = add_common_args(argparse.ArgumentParser()); a = ap.parse_args()
    dev, out, log = setup(a, "v4_checks"); tr, ca, te, info = get_data(a, log)
    for steps in (300, 3000, 10000):
        t0 = time.time(); seg = train_seg(tr, info, dev, steps=steps)
        log(f"judge trained {steps:6d} steps ({time.time() - t0:.0f}s): label error on TRUE colour  calibration {seg_err(seg, ca, dev):.3f}  test {seg_err(seg, te, dev):.3f}")

if __name__ == "__main__":
    main()

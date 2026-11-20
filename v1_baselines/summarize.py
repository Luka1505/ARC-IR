"""Mean +- std over seeds for every run folder pattern.   python -m v1_baselines.summarize
Looks at runs/b_<method>_<inputs>_s<seed> (baselines) and runs/arc_s<seed> (ARC-IR) and prints one table."""
import glob, json, os, re, numpy as np
G = {}
for f in sorted(glob.glob("runs/*/results.json")):
    d = os.path.basename(os.path.dirname(f)); m = re.match(r"(b_\w+?)_s(\d+)$", d) or re.match(r"(arc)_s(\d+)$", d)
    if m: G.setdefault(m.group(1), []).append(json.load(open(f)))
cols = [("sr_psnr", "SR PSNR"), ("sr_ssim", "SR SSIM"), ("chroma_mae", "chroma MAE"), ("flip_full_coverage", "class-flip"), ("label_error_seg_read", "label err (segmenter reads colour)"), ("label_error_head", "label err (class head)")]
print(f"{'run':22s} {'seeds':>5s} " + " ".join(f"{n:>34s}" for _, n in cols))
for k, rs in G.items():
    cells = []
    for key, _ in cols:
        v = [r[key] for r in rs if key in r]
        cells.append(f"{np.mean(v):.4f} +- {np.std(v):.4f}" if v else "-")
    print(f"{k:22s} {len(rs):5d} " + " ".join(f"{c:>34s}" for c in cells))

"""The frozen evaluation judge (primary) and a second, independently trained judge (used only by the optional task-driven variant).
Both are trained ONCE by v6_final/make_judge.py and loaded by every run, so every method is graded by the identical network."""
import os, torch
from .models import Seg
from .utils import train_seg


def load_judges(judge_dir, tr, info, dev, steps=3000):
    if judge_dir and os.path.exists(os.path.join(judge_dir, "judge.pt")):
        js = []
        for f in ("judge.pt", "judge2.pt"):
            s = Seg(info["K"]).to(dev); s.load_state_dict(torch.load(os.path.join(judge_dir, f), map_location=dev))
            for p in s.parameters(): p.requires_grad_(False)
            js.append(s.eval())
        return js
    if judge_dir:
        raise FileNotFoundError(f"{judge_dir}/judge.pt not found: run  python -m v6_final.make_judge  first (all methods must share one judge)")
    return [train_seg(tr, info, dev, steps=steps, seed=123), train_seg(tr, info, dev, steps=steps, seed=777)]     # offline tests only

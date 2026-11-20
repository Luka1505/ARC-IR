"""Does the HAT backbone build, run and fit on the GPU?   python -m v2_structure.test_hat [--weights HAT-L_SRx2_ImageNet-pretrain.pth] [--small]
Builds the adapter, runs one forward/backward on a 64x64 thermal tile (x2 -> 128x128) and prints parameters, shape, time and GPU memory."""
import argparse, time, torch
from common.models import build_structure
ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=None); ap.add_argument("--small", action="store_true"); ap.add_argument("--bs", type=int, default=8); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
st = build_structure("hat-s" if a.small else "hat", 2, a.weights).to(dev)
print(f"HAT{'-S' if a.small else '-L'}: {sum(p.numel() for p in st.parameters()) / 1e6:.1f} M parameters, device {dev}, weights {'loaded' if a.weights else 'RANDOM (architecture test only)'}")
x = torch.rand(a.bs if dev == "cuda" else 1, 1, 64, 64, device=dev)
t0 = time.time(); f, y = st(x); y.mean().backward()
if dev == "cuda": torch.cuda.synchronize()
print(f"forward+backward batch {x.shape[0]}: output {tuple(y.shape)} in {time.time() - t0:.1f} s" + (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev == "cuda" else ""))
assert y.shape[-2:] == (128, 128), "expected 2x super-resolution"; print("HAT adapter OK")

"""Smoke-test split from WHATEVER is downloaded so far (the real regions.yaml puts test/calibration regions last, so they are
not downloaded yet).  Writes data_tools/regions_smoke.yaml:  python -m data_tools.make_smoke_split
Only for checking that the code runs and gives sensible numbers. Never report these results in the paper."""
import glob, os, sys, yaml

d = sys.argv[1] if len(sys.argv) > 1 else "dataset"
have = sorted(r for r in os.listdir(d) if os.path.isdir(os.path.join(d, r)) and len(glob.glob(os.path.join(d, r, "[!_]*.npz"))) >= 20)
if len(have) < 3:
    sys.exit(f"only {len(have)} regions with >= 20 tiles so far ({have}); need at least 3. Wait for the download.")
n_cal = max(1, round(len(have) * 0.2)); n_test = max(1, round(len(have) * 0.2))
test = have[-n_test:]; cal = have[-(n_test + n_cal):-n_test]; train = have[:-(n_test + n_cal)]
yaml.safe_dump({"train": train, "calibration": cal, "test": test}, open("data_tools/regions_smoke.yaml", "w"))
print("train      :", train); print("calibration:", cal); print("test       :", test)
print("wrote data_tools/regions_smoke.yaml")

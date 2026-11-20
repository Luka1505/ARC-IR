import argparse, numpy as np
from scipy.stats import spearmanr
from common.utils import add_common_args, setup
from common.evalx import predict
from v4_abstain.checks import load

def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--ckpt", default="runs/v3_fixed/arc.pt"); ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args(); dev, out, log = setup(a, "v4_checks")
    model, ck, tr, ca, te, info = load(a, dev, log)
    P = predict(model, te, dev, ck["use_meta"], S=a.samples)
    err = (P["ab"] - te["ab"]).abs().mean(1).numpy(); sp, en = P["spread"].numpy(), P["ent"].numpy()
    reg = te["reg"].numpy(); rn = info.get("reg_names", {}).get("test", []); rng = np.random.default_rng(0)
    def row(name, m):
        e, s, n = err[m].ravel(), sp[m].ravel(), en[m].ravel()
        i = rng.choice(len(e), min(len(e), 400000), replace=False)
        rs, rn_ = spearmanr(s[i], e[i])[0], spearmanr(n[i], e[i])[0]
        lo_s, lo_n = e[s <= np.median(s)].mean(), e[n <= np.median(n)].mean()
        log(f"  {name:22s} spearman(spread,err) {rs:+.3f} | spearman(entropy,err) {rn_:+.3f} | colour err all {e.mean():.4f}, lowest-50% spread {lo_s:.4f}, lowest-50% entropy {lo_n:.4f}")
    log("does a score predict COLOUR error (chroma MAE per pixel)? positive correlation and lower error in the lowest-50% = useful")
    row("ALL test", np.ones(len(reg), bool))
    for r in np.unique(reg): row(rn[r] if r < len(rn) else f"test{r}", reg == r)

if __name__ == "__main__":
    main()

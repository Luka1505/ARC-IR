"""Runs the download in N parallel parts and RESTARTS any part that goes quiet (a hung network read prints nothing for hours).
    python -m data_tools.supervise                 # 4 parts, restart a part after 15 minutes without output
    python -m data_tools.supervise --parts 4 --stall-min 15
Finished scenes are never repeated, so a restart only loses the scene that was hanging. Stop everything with Ctrl+C."""
import argparse, subprocess, sys, threading, time, glob, os


EXTRA = []   # extra make_tiles arguments (--regions, --out), set from the command line
MODULE = ["data_tools.make_tiles"]


def run_part(i, n, stall, state, cmd=None):
    while not state["stop"]:
        c = cmd or [sys.executable, "-u", "-m", MODULE[0], "--part", f"{i}/{n}"] + EXTRA
        p = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last = [time.time()]; finished = [False]

        def reader():
            for line in p.stdout:
                last[0] = time.time(); line = line.rstrip()
                if "finished" in line: finished[0] = True
                if line.strip().startswith(("LC0", "LE0", "[", "WorldCover FAILED")) or "FAILED" in line:
                    print(f"[part {i}] {line[:160]}", flush=True)
        t = threading.Thread(target=reader, daemon=True); t.start()
        while p.poll() is None and not state["stop"]:
            time.sleep(5)
            if p.poll() is None and time.time() - last[0] > stall:
                print(f"[part {i}] no output for {stall / 60:.0f} min -> restarting it (done scenes are kept)", flush=True)
                p.kill(); break
        if state["stop"]:
            p.kill(); return
        p.wait(); t.join(timeout=5)
        if finished[0] and p.returncode == 0:
            print(f"[part {i}] finished", flush=True); return
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", type=int, default=4); ap.add_argument("--stall-min", type=float, default=15)
    ap.add_argument("--out", default="dataset"); ap.add_argument("--regions", default=None)
    ap.add_argument("--module", default="data_tools.make_tiles", help="or data_tools.backfill_kelvin / data_tools.backfill_color"); a = ap.parse_args(); MODULE[0] = a.module
    EXTRA.extend(["--out", a.out] + (["--regions", a.regions] if a.regions else []))
    state = {"stop": False}
    ts = [threading.Thread(target=run_part, args=(i, a.parts, a.stall_min * 60, state)) for i in range(1, a.parts + 1)]
    for t in ts: t.start()
    try:
        while any(t.is_alive() for t in ts):
            time.sleep(60)
            n = len(glob.glob(os.path.join(a.out, "*", "_kelvin_*" if "kelvin" in a.module else "_color_*" if "color" in a.module else "_done_*")))
            print(f"--- {time.strftime('%H:%M')}  scenes done so far: {n}", flush=True)
    except KeyboardInterrupt:
        state["stop"] = True; print("stopping...")
    for t in ts: t.join()


if __name__ == "__main__":
    main()

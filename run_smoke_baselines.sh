#!/bin/bash
# Baselines + ARC-IR seeds on the SMOKE split (same split, same inputs, same steps). Run from ~/arcir with the venv active:   bash run_smoke_baselines.sh
# Uses GPUs 0-3 only (the DGX is shared). Each batch of 4 jobs runs at the same time, then the next batch starts. Logs go to logs/.
mkdir -p logs
COMMON="--data landsat --regions data_tools/regions_smoke.yaml --steps 3000 --bs 8 --soft --terrain --dates 3 --abs-temp --color-target fixed"
run() { g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@" > logs/$(echo "$@" | tr -c 'a-zA-Z0-9\n' '_' | tail -c 80).log 2>&1 & }
# batch 1: input-matched baselines, seed 0
run 0 python -m v1_baselines.run_fair $COMMON --method reg      --inputs full    --seed 0 --out runs/b_reg_full_s0
run 1 python -m v1_baselines.run_fair $COMMON --method pix2pix  --inputs full    --seed 0 --out runs/b_pix2pix_full_s0
run 2 python -m v1_baselines.run_fair $COMMON --method reg      --inputs thermal --seed 0 --out runs/b_reg_thermal_s0
run 3 python -m v1_baselines.run_fair $COMMON --method sr       --inputs full    --seed 0 --out runs/b_sr_full_s0
wait
# batch 2: pix2pix thermal-only, and two more seeds of ARC-IR (seed 0 is runs/v3_fixed)
run 0 python -m v1_baselines.run_fair $COMMON --method pix2pix  --inputs thermal --seed 0 --out runs/b_pix2pix_thermal_s0
run 1 python -m v3_chroma.run $COMMON --init runs/v2_structure/structure.pt --freeze-st --seed 1 --out runs/arc_s1
run 2 python -m v3_chroma.run $COMMON --init runs/v2_structure/structure.pt --freeze-st --seed 2 --out runs/arc_s2
run 3 python -m v1_baselines.run_fair $COMMON --method reg      --inputs full    --seed 1 --out runs/b_reg_full_s1
wait
# batch 3: remaining baseline seeds
run 0 python -m v1_baselines.run_fair $COMMON --method reg      --inputs full    --seed 2 --out runs/b_reg_full_s2
run 1 python -m v1_baselines.run_fair $COMMON --method pix2pix  --inputs full    --seed 1 --out runs/b_pix2pix_full_s1
run 2 python -m v1_baselines.run_fair $COMMON --method pix2pix  --inputs full    --seed 2 --out runs/b_pix2pix_full_s2
run 3 python -m v3_chroma.run $COMMON --init runs/v2_structure/structure.pt --freeze-st --seed 0 --out runs/arc_s0
wait
python -m v1_baselines.summarize

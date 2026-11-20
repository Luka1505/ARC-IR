#!/bin/bash
# FINAL RUN launcher. Usage (from ~/arcir, env active):   bash run_final_all.sh judge | timing | launch
# Golden rules: same split (regions_big.yaml), same judge (runs/final_judge), seeds 0-2, 30,000 steps per trained stage, 8-fold TTA everywhere.
cd ~/arcir && source ~/arcir_env/bin/activate && mkdir -p logs
DATA="--data landsat --tiles-dir dataset_big --regions data_tools/regions_big.yaml --soft --terrain --dates 3 --abs-temp --color-target fixed"
RUN="$DATA --bs 8 --w-edge 0.05 --judge-dir runs/final_judge"
case "$1" in
judge)   CUDA_VISIBLE_DEVICES=0 python -m v6_final.make_judge $DATA --out runs/final_judge 2>&1 | tee logs/final_judge.log ;;
timing)  CUDA_VISIBLE_DEVICES=0 python -m v6_final.run_final $RUN --method arc --steps 300 --stage1-steps 300 --no-save-preds --out runs/final_timing 2>&1 | tee logs/final_timing.log ;;
launch)
  S=30000
  lane() { g=$1; shift; ( for job in "$@"; do m=${job%:*}; s=${job#*:}; x=""; [ "$m" = arc_task ] && { m=arc; x="--task-w 0.1"; }
      n=$m; [ -n "$x" ] && n=arc_task; CUDA_VISIBLE_DEVICES=$g python -m v6_final.run_final $RUN --method $m --seed $s --steps $S --stage1-steps $S $x > logs/final_${n}_s$s.log 2>&1; done ) & }
  lane 0 arc:0 reg:0 pix2pix:0
  lane 1 arc:1 reg:1 pix2pix:1
  lane 2 arc:2 reg:2 pix2pix:2
  lane 3 arc_task:0 sr:0 sr:1 sr:2 regvar:0
  wait ;;
*) echo "use: judge | timing | launch" ;;
esac

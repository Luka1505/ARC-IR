#!/usr/bin/env bash
# Run from the arcir folder. Downloads in the background into ./dataset and can be stopped/restarted safely.
set -e
mkdir -p dataset
python -m data_tools.check_access
nohup python -m data_tools.make_tiles > dataset/download.log 2>&1 &
echo "started. watch it with:  tail -f dataset/download.log     status: python -m data_tools.make_tiles --summary"

# ARC-IR code, staged in versions

Each version is its own folder with its own `run.py`. Shared code lives in `common/` and changes rarely.
Run everything from this folder with `python -m <folder>.run ...`.

| Version | Folder | Adds | Question it answers | Needs |
|---|---|---|---|---|
| v1 | `v1_baselines` | bicubic SR, regression colorizer (+ metadata control) | What number must ARC-IR beat? | data only |
| v2 | `v2_structure` | structure branch (SR) | Does our SR beat bicubic / pretrained backbone? | data |
| v3 | `v3_chroma` | material head + flow-matching colour | Does colour generation work at full coverage? | v2 checkpoint (optional) |
| v4 | `v4_abstain` | ambiguity score + calibrated threshold | Does abstaining cut wrong colours? (no training) | v3 checkpoint |
| v5 | `v5_full` | joint training + multi-date invariance + ablation table | The paper's main results table | data (v2 optional) |

## Try it now (synthetic data, no downloads)
```
pip install -r requirements.txt
python -m v1_baselines.run --steps 300
python -m v2_structure.run --steps 600
python -m v3_chroma.run --steps 600 --init runs/v2_structure/structure.pt --freeze-st
python -m v4_abstain.run --ckpt runs/v3_chroma/arc.pt --alpha 0.05
python -m v5_full.run --steps 600 --seeds 1
python -m common.test_data_tools        # checks the real-data tile format offline
```
Every run prints `[timer] ... s/step -> N steps ~ X min`. Run 50 steps first on the GPU, read that line, then launch the long run.

## Getting the dataset (do this first)
Folder layout: unzip so you have `arcir/`, and all data goes in `arcir/dataset/` (created automatically).
```
cd arcir
pip install -r requirements.txt
python -m data_tools.check_access                                   # 1. can this machine reach the data? (4 checks)
python -m data_tools.make_tiles --only krishna_delta --max-scenes 1  # 2. ONE scene first; fix anything that fails
python -m data_tools.preview --region krishna_delta                  # 3. open dataset/preview_krishna_delta.png and look:
                                                                     #    thermal | colour | land cover must line up
./start_download.sh                                                  # 4. everything, in the background, resumable
tail -f dataset/download.log ; python -m data_tools.make_tiles --summary
```
Size: 30 regions x up to 30 scenes, 128x128 tiles. Expect very roughly 80-130k tiles, about 25-40 GB on disk (an estimate: cloud cover and scene edges change it). Loading tiles takes about 0.3 MB of RAM each, so on the 32 GB PC always train with `--max-tiles 40000` (about 12 GB); use all tiles on the DGX.
Train on it with `--data landsat` (the loader reads `dataset/` by default), e.g. `python -m v1_baselines.run --data landsat --steps 3000 --bs 8`.

## Switching to real Landsat data
1. Regions: `data_tools/regions.yaml` (30 regions, each in exactly one of train / calibration / test; coordinates are approximate, check the preview).
2. Download: see "Getting the dataset" above.
3. Add `--data landsat` to any command above (defaults: `--tiles-dir dataset --regions data_tools/regions.yaml`).
4. Pretrained backbone: `--structure hat --weights path/to/HAT-L.pth` (needs `basicsr`; untested here, check HAT's config against your checkpoint).

## What is tested and what is not
Tested (synthetic data, CPU): every version runs end to end; tile writer -> loader -> model on fake scenes.
Not tested: the live Planetary Computer download and WorldCover fetch (`check_access` and the one-scene test exist to find problems early), the HAT adapter, anything on GPU.
Results on synthetic data only show the code works; they say nothing about Landsat.

## When to change what
`common/` (data format, metrics, models) should stay stable. A new idea = a new version folder or a new flag, so earlier versions keep reproducing.

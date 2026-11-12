# train/ — ternary MNIST models (BitNetMCU)

Phase 0, step 3: reproduce BitNetMCU's ternary MNIST model, plus sparse variants for the sparse-storage experiment.

## Setup (Windows, Python 3.12, CUDA GPU)
```
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install numpy PyYAML tensorboard matplotlib torchsummary seaborn
git clone https://github.com/cpldcpu/BitNetMCU.git BitNetMCU
```
`torchsummary` and `seaborn` are imported by BitNetMCU but missing from its `requirements.txt`.

## Train / export / inspect (run from `train/BitNetMCU`)
```
../.venv/Scripts/python.exe training.py    --params ../configs/ternary_cnn80.yaml
../.venv/Scripts/python.exe exportquant.py --params ../configs/ternary_cnn80.yaml   # writes BitNetMCU_model.h -> move to ../export/
../.venv/Scripts/python.exe ../sparsity_report.py ../configs/ternary_cnn80.yaml modeldata/<runname>.pth
```
About 24 s per epoch on an RTX 3060 Ti, so a 60-epoch run takes about 25 min. Export reads `modeldata/<runname>.pth`, which is the final epoch, not `_best`.

## Configs
| Config | fc1 zeros | Notes |
|---|---|---|
| `ternary_cnn80.yaml` | 50.2% (natural, BitNet absmean rule) | BitNetMCU's "80-wide ternary" run; they report 99.52% |
| `ternary_cnn80_z70.yaml` | 70% (forced) | sparse-storage variant |
| `ternary_cnn80_z90.yaml` | 90% (forced) | sparse-storage variant |
| `int8_cnn80.yaml` | 5% (natural) | the same CNN with every linear layer int8 (`8bit`), for the int8 compiler test (FINDINGS F24) |

Model: 8-bit depthwise CNN (80 ch) → **fc1 320→96 ternary** → fc2 96→64 4-bit → classifier 64→10 4-bit. Only fc1 is ternary. It holds 30,720 of the 39,664 weights. Total size 11.42 kB.

## Local patches to BitNetMCU (marked `model-as-firmware patch`)
- `models.py` `CNNMNIST`: new `fc1_quant` arg (upstream hardcodes `'2bitsym'`) and `ternary_zero_frac` arg.
- `BitNetMCU.py` `weight_quant`: when `ternary_zero_frac > 0`, a Ternary layer zeroes exactly that fraction of its smallest-|w| weights. The remaining weights become ±1, scaled by the mean of the kept |w|.
- `exportquant.py` `load_model`: forwards `fc1_quant` and `ternary_zero_frac`. Without this, upstream export silently re-quantized a ternary fc1 as 2-bit.

The clone isn't tracked in this repo. The patches are saved in `bitnetmcu.patch`, made against upstream commit `0715bfc`. To apply them after a fresh clone:
```
git -C BitNetMCU checkout 0715bfc4ed9f2578496e17b4b0e13f2297e3cc0f
git -C BitNetMCU apply ../bitnetmcu.patch
```

## Results
| Model | fc1 zeros | Best test acc | Final test acc | Quantized (integer) acc |
|---|---|---|---|---|
| tern80 | 50.2% | 99.48% (ep 52) | 99.45% | 99.47% |
| tern80z70 | 70.0% | 99.55% (ep 58) | 99.44% | 99.45% |
| tern80z90 | 90.0% | 99.54% (ep 59) | 99.50% | 99.49% |
| int8_80 (all linear int8) | 5% | 99.50% (ep 45) | 99.43% | 99.45% |

Even with 90% of fc1's weights set to zero, accuracy doesn't drop; all the differences are within run-to-run noise. Final-epoch weights are in `models/*.pth`, and the exported C headers are in `export/*.h`. Note that the packed ternary format is a fixed 1.6 bits per weight however many zeros there are, so the size savings will only come from a sparse format in the compiler.

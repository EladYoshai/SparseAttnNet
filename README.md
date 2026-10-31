# SparseAttnNet

Code for the paper **"Hierarchical Sparse Attention Framework for Computationally Efficient Classification of
Biological Cells"** (E. Yoshai, D. Yagoda-Aharoni, E. Dotan, N. T. Shaked).

SparseAttnNet classifies a cell image from a small, learned subset of its pixels. A light coarse convolutional
network scores every pixel. The k highest-scoring pixels are embedded as points (position and intensity) and
summarized by a fine attention with linear complexity in k. The result is fused with pooled coarse features for
classification. The fine attention teaches the coarse network where to look through a distillation loss, and k is
adapted during training from the training loss.

## Contents

| File | Purpose |
|---|---|
| `core.py` | model, data loading, losses, metrics, baselines |
| `run.py` | nested 5-fold cross-validation of one model or ablation on one dataset |
| `run_all.sh` | runs every experiment of the paper in order (resumable) |
| `paired_stats.py` | paired comparisons over folds: t-test, Nadeau-Bengio corrected t-test, Wilcoxon, 95 % CI |
| `deletion_test.py` | faithfulness test (deletion and insertion of the selected pixels) |
| `benchmark.py` | FLOPs per stage, parameters, CPU and GPU latency, throughput, memory |
| `flops_by_k.py` | FLOPs of SparseAttnNet for each pixel budget k |
| `fetch_weights.py` | downloads and checks (SHA-256) the ImageNet weights of the pretrained transformer baselines |
| `make_figures.py` | accuracy versus FLOPs, confusion matrices, reliability diagrams, deletion curves, k trajectories and ablation |
| `summarize.py` | prints the result tables (classification metrics, per-class results, ablations) |

## Installation

Python 3.12 and a CUDA GPU (the paper used one NVIDIA RTX 5070 Ti, 16 GB).

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
```

## Data

See [`data/README.md`](data/README.md). The WBC dataset is public (Kaggle). The event-camera, raw-hologram and
OPD datasets were acquired in our lab and are not included. They are available from the corresponding author on
reasonable request.

## Reproducing the results

Everything at once (each finished job is skipped on a restart):

```bash
bash run_all.sh          # all four datasets
bash run_all.sh wbc      # only the public WBC dataset
```

Single runs:

```bash
python run.py --dataset wbc --model ours --fast                          # SparseAttnNet
python run.py --dataset wbc --model resnet50 --fast                      # baselines: resnet50, mobilenet,
                                                                         #   ConvNeXt, DeiT, ViTSmall, SwinTiny
python fetch_weights.py                                                  # ImageNet weights (pretrained runs only)
python run.py --dataset event --model DeiT --pretrained --fast --tag DeiT_pretrained
python run.py --dataset wbc --model ours --fixed_k 4000 --fast --tag fixed_k4000             # ablations
python run.py --dataset wbc --model ours --random_select --k_from ours --fast --tag random_matched
python paired_stats.py; python summarize.py
```

`--fast` keeps the data on the GPU. It gives the same batches and the same arithmetic as the standard data loader
and is only faster. `--folds 0 --max_epochs 1` gives a quick smoke test.

### Protocol

* **Outer folds.** Stratified 5-fold cross-validation. For the flow-imaging datasets (event camera, holograms,
  OPD) the folds are contiguous blocks in acquisition order, so all projections of one physical cell stay in the
  same fold. For WBC (independent images) the folds are shuffled with seed 123456.
* **Inner validation.** 20 % of each outer-training portion (the last 20 % of each class in acquisition order;
  a stratified random 20 % for WBC). It is used for checkpoint selection and the learning-rate schedulers.
* **Training budget.** Every model, SparseAttnNet and baselines alike, trains for exactly 200 epochs with no
  early stopping and no data augmentation (resize only). The checkpoint with the best inner-validation accuracy is
  kept, and the outer test fold is evaluated once, after training, with that checkpoint (`TestGuard` in `run.py`
  raises an error otherwise).
* **Settings.** SparseAttnNet: AdamW (learning rate 1e-3, weight decay 1e-4), focal loss (gamma 2), contrastive
  loss (weight 0.1, temperature 0.07), distillation (weight 0.1, emphasis 2, 5 warm-up epochs), initial k 3000
  (WBC 8000), minimum k 1500, input 135 x 135 (WBC 256 x 256), batch 64 (WBC 32). ResNet50 and MobileNetV2: same
  input size, learning rate 1e-3. Transformers (ConvNeXt-Tiny, DeiT-Small, ViT-Small, Swin-Tiny): 224 x 224,
  learning rate 5e-5. All values are in `core.Config`.
* **Seeds.** Seed 123456 plus the fold index for every model.

### Outputs

`results/<dataset>/<run>/summary.json` holds the mean and standard deviation over the five test folds (accuracy,
weighted and macro precision, sensitivity, specificity, F1, expected calibration error, selected k).
`fold_<i>/` holds the test predictions, the metrics, the training history and the data split of each fold.
`results/paired_stats.md`, `results/faithfulness/`, `results/benchmark/` and `figures/` hold the remaining
tables and figures.

| Paper item | Produced by |
|---|---|
| Classification results and per-class results | `run.py`, then `summarize.py` |
| Paired statistical comparisons | `paired_stats.py` |
| Efficiency (FLOPs, parameters, latency, throughput, memory) and FLOPs per stage | `benchmark.py` |
| Ablation study | `run.py` ablation flags (`run_all.sh`), then `summarize.py` |
| Confusion matrices, reliability diagrams, deletion curves, accuracy versus FLOPs, k trajectories | `make_figures.py` |

Timing on one RTX 5070 Ti: a full 5-fold SparseAttnNet run takes about 10 to 30 minutes on the flow-imaging
datasets and about 2 hours on WBC. Baselines take up to several hours each, so the complete `run_all.sh` takes
a few days. Results can differ slightly on other GPUs or library versions because of nondeterministic GPU
kernels.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

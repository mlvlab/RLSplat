# RLSplat: Feed-Forward 3D Gaussian Splatting for Novel-View Synthesis

RLSplat reconstructs a 3D Gaussian scene from a sparse set of uncalibrated images and renders novel views. This repository provides evaluation and inference code for the released checkpoint. Gaussian rendering uses [gsplat](https://github.com/nerfstudio-project/gsplat).

Project page:<br>
Paper: [arXiv:2610.27712](https://arxiv.org/abs/2610.27712)<br>

Training code will be released in a future update.

## 🔧 Installation

The evaluation code is intended for Python 3.11 with CUDA and a compatible PyTorch installation. Install the PyTorch build that matches your CUDA toolkit, then install the remaining dependencies:

```bash
conda create -y -n rlsplat python=3.11
conda activate rlsplat

# Install the PyTorch and torchvision builds matching your CUDA version first.
pip install torch torchvision
pip install -r requirements.txt

# DVLT's package metadata may try to replace PyTorch; install its pinned source without dependencies.
pip install --no-deps \
  git+https://github.com/nv-tlabs/dvlt.git@f59bb18259ab446a7585c4f47731a77cfdefa426

# Build PyTorch3D against the PyTorch/CUDA environment above.
pip install --no-build-isolation \
  git+https://github.com/facebookresearch/pytorch3d.git
```

The first run downloads DVLT and LPIPS weights and may compile gsplat CUDA kernels. Set `TORCH_EXTENSIONS_DIR` to a writable directory if needed.

## 📦 Pre-trained Checkpoint

The released checkpoint is hosted on [Hugging Face](https://huggingface.co/Knowing/RLSplat).

| Checkpoint | Dataset | Training context views | Subfolder |
| --- | --- | ---: | --- |
| RLSplat | RE10K | Dynamic 2–24 (max 24) | `re10k-24view` |

Download the checkpoint:

```bash
huggingface-cli download Knowing/RLSplat \
  --repo-type model \
  --include "re10k-24view/*" \
  --local-dir checkpoints/rlsplat
```

The evaluation scripts can also load it directly from the Hub with `hf://Knowing/RLSplat/re10k-24view/model.safetensors`.

## 🎞️ Try Your Own Images

Pass 2–4 images of the same scene to render a video along the predicted camera path between the first and last input views:

```bash
python scripts/render_images.py view1.jpg view2.jpg view3.jpg \
  --budget-per-image 2048 \
  --frames 36 \
  --fps 12 \
  --output novel_view.mp4
```

`--budget-per-image` controls the number of Gaussians allocated per input image (default: 4,096). For example, 3 images with a budget of 2,048 use a total budget of 6,144 Gaussians. Use overlapping images with a clear viewpoint change. The script center-crops inputs to square; `--input-size`, `--render-size`, `--frames`, `--fps`, and `--render-batch-size` can also be adjusted.

## 📊 Evaluation

Prepare RE10K or ACID in the preprocessed, chunked `.torch` format. Set `DATA_ROOT` to the dataset root containing `test/index.json` and its data chunks. Fixed evaluation selections for 2, 8, 16, and 24 context views are provided in `assets/`.

For example, evaluate the checkpoint on ACID with 8 context views and the listed Gaussian budgets:

```bash
DATA_ROOT=/path/to/acid
OUTPUT_ROOT=analysis_outputs
CHECKPOINT=hf://Knowing/RLSplat/re10k-24view/model.safetensors
INDEX=assets/evaluation_index_acid_ctx_8.json

python -m src.main '+evaluation=acid' \
  "checkpointing.load=$CHECKPOINT" \
  "dataset.re10k.roots=[$DATA_ROOT]" \
  "dataset.re10k.view_sampler.index_path=$INDEX" \
  dataset.re10k.view_sampler.num_context_views=8 \
  "test.output_path=$OUTPUT_ROOT/acid_ctx8" \
  test.gaussian_budget_sweep=true \
  'test.gaussian_budget_per_view_list=[1024,2048,4096,8192,16384]' \
  wandb.name=acid_ctx8
```

For RE10K, use `'+evaluation=re10k'`, the RE10K dataset root, and `assets/evaluation_index_re10k_ctx_8.json`. The two-view indices are named `assets/evaluation_index_<dataset>.json`; other included indices use `assets/evaluation_index_<dataset>_ctx_<views>.json`. Set `test.align_pose=false` to disable pose alignment. Per-scene metrics and budget AUC are written under `test.output_path`. To aggregate a budget sweep, run:

```bash
python scripts/eval/aggregate.py analysis_outputs
```

## 🙏 Acknowledgements

We thank the authors and contributors of [gsplat](https://github.com/nerfstudio-project/gsplat) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for their open-source work.

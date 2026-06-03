# Hilbert Diffusion Model — 1D Function Generation

This repository is a codebase for **1D function generation** experiments based on the Hilbert Diffusion Model from Lim et al. (NeurIPS 2023), *Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces*. The current implementation focuses on:

- Generating synthetic 1D function datasets
- Resolution-free training / validation / sampling
- Comparing operator backbones: FNO / KNO / MHLKNO (Galerkin-type KNO, RFF only) / MHLKNO_LINATTN (Galerkin-type KNO)
- Evaluation metrics: Power (MMD-based two-sample test) and Moments (mean-function L2, covariance HS)

The runtime environment assumes a **single GPU** (e.g., Google Colab L4 GPU).

---

## Repository layout (core files)

- `main.py`: entrypoint for training/sampling (includes CLI argument parsing)
- `configs/*.yml`: experiment configuration files (YAML). Pass only the **file name** to `--config`; the loader reads from `./configs/<filename>`.
- `datasets/`: synthetic 1D datasets (`QuadraticDataset`) and `get_dataset()`
- `runner/hilbert_runner.py`: training loop, resolution-free validation, sampling (especially Tsit5 ODE), checkpoint loading
- `models/`: FNO and (MHL)KNO family models
- `functions/`: SDE definition, loss, samplers, Tsit5 probability-flow ODE sampler
- `evaluate/`: Power and Moments computation

---

## Installation (Colab)

The commands below match the current dependencies and execution flow in this codebase.

```bash
!pip install -r requirements.txt

# Prevent/resolve torchode conflicts (may be needed depending on the Colab runtime)
%pip uninstall -y torchtyping
!pip install git+https://github.com/martenlienen/torchode.git

# diffusers / transformers / accelerate
!pip install --upgrade "diffusers>=0.28" transformers accelerate

# (extra dependency)
!pip install sktime --quiet
```

`requirements.txt` includes the core packages (einops, scipy, tensorly, torchsde, torchode, tensorboard, etc.).

---

## Quickstart

### 1) Training

The command below runs training and then **automatically runs sampling once after training finishes** (the current default behavior of `main.py`).

```bash
!python main.py \
  --config hdm_quadratic_fno.yml \
  --doc quadratic_experiment \
  --exp outs \
  --res_free_points 100 200 300
```

- `--exp outs`: root directory for outputs
- `--doc quadratic_experiment`: experiment name (log folder name)
- `--res_free_points 100 200 300`: list of resolutions (numbers of points) used for resolution-free validation

### 2) Sampling — Tsit5 probability-flow ODE

To generate samples only from a trained checkpoint, enable `--sample`. The example below uses Tsit5 ODE sampling and loads the checkpoint at `ckpt_step=200`.

```bash
!python main.py \
  --config hdm_quadratic_fno.yml \
  --doc quadratic_experiment \
  --exp outs \
  --sample \
  --sample_type tsit5_ode \
  --nfe 500 \
  --ckpt_step 200 \
  --res_free_points 100
```

- `--sample_type tsit5_ode`: Tsit5-based probability-flow ODE sampler (`functions/tsit5_sampler.py`)
- `--nfe 500`: number of ODE solver inference steps (corresponds to function evaluations)
- `--ckpt_step 200`: first tries `outs/logs/quadratic_experiment/ckpt_step_200.pth`  
  (if missing, falls back to the latest `ckpt.pth` or `model.ckpt_dir` from the config)

---

## Outputs

With `--exp outs` and `--doc quadratic_experiment`:

- Logs / checkpoints: `outs/logs/quadratic_experiment/`
  - `config.yml`: snapshot of the config used for the run
  - `stdout.txt`: training logs
  - `ckpt.pth`: latest checkpoint
  - `ckpt_step_{step}.pth`: periodic checkpoints saved every `training.ckpt_store` steps
- TensorBoard: `outs/tensorboard/quadratic_experiment/`
- Sample plots: by default `outs/samples/images/`
  - For Tsit5 sampling, the code saves `tsit5_ground_truth.pdf` and `tsit5_sample.pdf`.
  - You can change the folder name via `--image_folder`; the resolved path becomes `outs/samples/<image_folder>`.

---

## Config guide (key settings)

Example config: `configs/hdm_quadratic_fno.yml`

### Data (`data.*`)

- `data.dataset`: choose the 1D function family  
  Supported: `"Quadratic" | "Linear" | "Circle" | "Sin" | "Sinc" | "Doppler" | "GaussianBumps"`
- `data.dimension`: default number of grid points `N` during training
- `data.grid_type`: `"uniform"` or `"random"`
  - If `"random"` in training mode, `__getitem__` samples new coordinates each time, encouraging resolution-independent learning.
- `data.hyp_len`, `data.hyp_gain`, `data.num_basis`: hyperparameters for Hilbert noise (SE-kernel eigendecomposition)

### Model (`model.*`)

- `model.model_type`: `"FNO" | "KNO" | "MHLKNO" | "MHLKNO_LINATTN"`
- `model.in_channels`: typically set to 2 (function values + normalized coordinates)
- FNO skeleton: `hidden_channels`, `lifting_channels`, `projection_channels`, `n_layers`, `n_modes`, `norm`, `skip`, etc.
- (KNO family) kernel / random-feature settings: `kernel_type`, `kernel_Q`, `enable_kernel_time_cond`, `rf_backend`, `rf_total_basis`, `rf_rff_frac`, `taylor_degree`, `num_kernel_heads`, etc.

### Diffusion / Training / Sampling / Optim

- `diffusion.num_diffusion_timesteps`: number of diffusion discretization steps (for training)
- `training.*`: batch size, epochs, validation/checkpoint frequency
- `sampling.batch_size`: batch size for sample generation
- `sampling.enable_rms_clip`, `sampling.rms_clip_threshold`: RMS clipping options for Tsit5 ODE sampling (applied to visualization/saved samples)
- `optim.*`: AdamW learning rate and gradient clipping, plus per-parameter-group LR scaling for KNO/MHLKNO kernel parameters

---

## CLI arguments (frequently used)

Run `python main.py --help` to see all options. Common ones:

- `--config`: YAML file name under `./configs`
- `--exp`: root output directory (default `exp`)
- `--doc`: log folder name (experiment name)
- `--sample`: run sampling only (skip training)
- `--sample_type`: `sde | srk | tsit5_ode | ...`  
  (**In the current 1D experiments, the sampler used in practice is `tsit5_ode`.**)
- `--nfe`: number of sampler steps (especially inference steps for Tsit5 ODE)
- `--ckpt_step`: load a specific step checkpoint
- `--res_free_points`: list of point counts for resolution-free validation/sampling
- `--disable_resolution_free`: disable resolution-free mode (validate/sample only on a fixed grid)
- `--distributed`: run with torch DDP (not needed for single-GPU experiments)

---

## Metrics

During sampling (especially Tsit5 ODE), the following metrics are printed to logs.

- **Power**: reports the rejection rate of an MMD-based two-sample test, averaged over multiple repeats with mean±CI format. (`evaluate/power.py`)
- **Moments**:
  - mean function L2 error
  - covariance operator Hilbert–Schmidt error  
  (`evaluate/moments.py`; integrals approximated with 1D trapezoidal weights)

---

## Citation

To cite the HDM paper this codebase is built on:

```bibtex
@inproceedings{
lim2023scorebased,
title={Score-based Generative Modeling through Stochastic Evolution Equations},
author={Lim, Sungbin and Yoon, Eunbi and Byun, Taehyun and Kang, Taewon and Kim, Seungwoo and Lee, Kyungjae and Choi, Sungjoon},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=GrElRvXnEj}
}
```

# Self-Supervised Pansharpening

Two-phase self-supervised pansharpening framework with sparse coding (CSC) fusion.

## File Structure

```
zup_self/
  train_self.py        Main entry point (CLI)
  config.py            Hyperparameters, CSC config, dataset paths
  trainer.py           Phase1 Wald pretrain + Phase2 self-distillation
  data.py              HDF5 dataset loader
  mymodel.py           SDNetFusionNet models (All-CSC / Conv ablation)
  loss.py              Spatial fidelity & spectral loss functions
  wald_utilities.py    Wald protocol: MTF generation, 23-tap interpolation
  requirements.txt     Python dependencies
```

## Environment

```bash
conda create -n zspan python=3.10
conda activate zspan
pip install -r requirements.txt
```

PyTorch with CUDA support (recommended):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Dataset

The code reads HDF5 (`.h5`) files with the following keys per image:

| Key   | Description              |
|-------|--------------------------|
| `ms`  | Low-resolution MS        |
| `lms` | Upsampled LRMS           |
| `pan` | Panchromatic             |
| `gt`  | Ground truth (optional)  |

Default paths are configured in `config.py` under `DATASET_PRESETS`. Supported sensors: WV3, WV2, QB, GF2.

## Usage

### Full two-phase training (default)

```bash
python train_self.py --mode full --dataset WV3
```

Processes all 20 images (0–19) sequentially.

### Single image

```bash
python train_self.py --mode full --dataset WV3 --data_id 0
```

### Phase 1 only (LR Wald pretrain)

```bash
python train_self.py --mode lr --dataset WV3
```

### Ablation experiments

| Flag                      | Effect                                |
|---------------------------|---------------------------------------|
| `--ablation no_csc`       | Replace CSC layers with plain Conv    |
| `--ablation no_spatial`   | Remove spatial fidelity loss          |
| `--ablation no_spectral`  | Remove spectral loss                  |
| `--ablation no_phase1`    | Skip variance (teacher) loss          |
| `--ablation fix_lmbd`     | Freeze all lmbd parameters            |
| `--fista_T 5`             | Override FISTA iteration count        |
| `--lmbd_weight 300`       | Override lambda regularization weight |

Example:

```bash
python train_self.py --mode full --dataset WV3 --fista_T 5
python train_self.py --mode full --dataset WV3 --lmbd_weight 300
```

### Full CLI reference

```
--mode        {lr, full}          Training mode (default: full)
--dataset     {WV3, WV2, QB, GF2} Sensor type (default: WV3)
--data_id     INT                 Single image index (default: all)
--start_id    INT                 Start index for batch (default: 0)
--end_id      INT                 End index for batch (default: 19)
--device      STR                 Device (default: cuda:0)
--lr_phase1   FLOAT               Phase 1 learning rate
--epochs_phase1 INT               Phase 1 epochs
--lr_phase2   FLOAT               Phase 2 learning rate
--epochs_phase2 INT               Phase 2 epochs
--batch_size  INT                 Batch size
--w_spa       FLOAT               Spatial loss weight
--w_spec      FLOAT               Spectral loss weight
--lmbd_weight FLOAT               Lambda L2 regularization weight
--fista_T     INT                 FISTA steps override
--ablation    STR                 Ablation mode
```

## Output

Results are saved under:

- `result_self/{DATASET}_{mode}/` — `.mat` result files per image
- `model_FUG/` — saved model checkpoints (`.pth`)

Each output `.mat` contains: `I_MS_LR`, `I_MS`, `I_PAN`, `proposed`, and optionally `gt`.

## Key Hyperparameters

| Parameter        | Default    | Description                    |
|------------------|------------|--------------------------------|
| `lr_phase1`      | 0.015      | Phase 1 learning rate          |
| `epochs_phase1`  | 240        | Phase 1 training epochs        |
| `lr_phase2`      | 0.01       | Phase 2 learning rate          |
| `epochs_phase2`  | 2000       | Phase 2 training epochs        |
| `w_spa`          | 200        | Spatial fidelity loss weight   |
| `w_spec`         | 250        | Spectral loss weight           |
| `w_var`          | 1,850,000  | Variance (teacher) loss weight |
| `lmbd_weight`    | 500        | Lambda L2 regularization       |
| `ratio`          | 4          | PAN/MS resolution ratio        |
| `max_value`      | 2047       | 11-bit data normalization      |
| `seed`           | 10         | Random seed                    |

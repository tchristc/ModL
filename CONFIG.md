# ModL Training Configuration Guide

This file explains all parameters in `config.json` for training the ModL model.

## Dataset Paths

- **ProcessedDir**: Root directory containing preprocessed models (output of `modl preprocess`)
- **TrainIndexFile**: Text file listing training model IDs (one per line). Set by `modl split`
- **ValIndexFile**: Validation split index file
- **TestIndexFile**: Test split index file
- Set index files to `null` to use all data for training (not recommended)

## Model Architecture

- **NumClasses**: Number of output categories
  - ModelNet10: `10`
  - ModelNet40: `40`
  - ShapeNetCore55: `55`
  - **Must match your dataset!**

- **VoxelLatentDim**: Dimension of voxel encoder output (default: `256`)
- **ViewLatentDim**: Dimension of multi-view encoder output (default: `256`)
- **EmbeddingDim**: Fused embedding dimension before classifier (default: `512`)
- **Dropout**: Dropout probability (0.0–1.0, default: `0.4`)

## Input Preprocessing

- **VoxelResolution**: Voxel grid size (e.g., `32`, `64`, `128`)
  - **Must match** the resolution used in `modl preprocess`
- **NumViews**: Number of rendered views per model (default: `12`)
  - **Must match** preprocessing
- **ViewImageSize**: Images resized to this H×W before batching (default: `128`)

## Training Hyperparameters

- **Epochs**: Total training epochs (default: `100`)
- **BatchSize**: Models per batch (default: `32`)
  - Reduce if out-of-memory
  - Increase for faster training with more VRAM
- **LearningRate**: Initial learning rate (default: `0.001` = 1e-3)
- **WeightDecay**: L2 regularization (default: `0.0001` = 1e-4)

## Learning Rate Schedule

- **LrSchedule**: `"cosine"` or `"step"`
  - `"cosine"`: Smooth decay to 0 over all epochs (recommended)
  - `"step"`: Multiply LR by `LrStepGamma` every `LrStepEvery` epochs

- **LrStepGamma**: Decay factor for step schedule (default: `0.5`)
- **LrStepEvery**: Apply decay every N epochs (default: `20`)

## Data Augmentation

- **Augment**: Enable random rotations/flips/scales during training (default: `true`)

## Hardware

- **Device**: `"cpu"`, `"cuda"`, or `"cuda:0"`, `"cuda:1"`, etc.
  - Set to `"cuda"` if you have an NVIDIA GPU with CUDA installed
  - Will auto-fallback to CPU if CUDA unavailable

## Checkpointing

- **CheckpointDir**: Where to save model weights (default: `"checkpoints"`)
- **ResumeFrom**: Path to checkpoint directory to resume from (e.g., `"checkpoints/epoch_0050"`)
  - Set to `null` to start fresh
- **SaveEveryEpochs**: Save periodic checkpoints every N epochs (default: `10`)
  - Set to `0` to only save the best model

---

## Example Configurations

### Quick Test (ModelNet10)
```json
{
  "ProcessedDir": "C:\\data\\modelnet10_processed",
  "TrainIndexFile": "C:\\data\\modelnet10_processed\\train.txt",
  "ValIndexFile": "C:\\data\\modelnet10_processed\\val.txt",
  "NumClasses": 10,
  "VoxelResolution": 32,
  "Epochs": 20,
  "BatchSize": 16,
  "Device": "cpu"
}
```

### Full ModelNet40 Training (GPU)
```json
{
  "ProcessedDir": "C:\\data\\modelnet40_processed",
  "TrainIndexFile": "C:\\data\\modelnet40_processed\\train.txt",
  "ValIndexFile": "C:\\data\\modelnet40_processed\\val.txt",
  "NumClasses": 40,
  "VoxelResolution": 64,
  "NumViews": 12,
  "Epochs": 100,
  "BatchSize": 32,
  "LearningRate": 0.001,
  "LrSchedule": "cosine",
  "Device": "cuda",
  "CheckpointDir": "C:\\checkpoints\\modelnet40"
}
```

### Resume Training
```json
{
  "ProcessedDir": "C:\\data\\processed",
  "NumClasses": 40,
  "ResumeFrom": "C:\\checkpoints\\epoch_0030",
  "Epochs": 100
}
```

---

## Usage

```powershell
# Train with this config
modl train config.json

# Train with GPU override
modl train config.json --gpu

# Resume from checkpoint
modl train config.json --resume C:\checkpoints\epoch_0050

# Override epoch count
modl train config.json --epochs 200
```

---

## Notes

- **Architecture parameters** (`NumClasses`, latent dims, resolution, views) **cannot** be changed when resuming from a checkpoint — they must match exactly
- The trainer automatically saves `config.json` to `CheckpointDir` for reproducibility
- Use `modl evaluate` and `modl export` with the same checkpoint — they auto-load the saved config

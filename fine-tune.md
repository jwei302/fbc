<!-- 38b804eb-f373-4016-af2b-3dec9ff2b67b 40a636ca-4dd2-4087-8f42-b974be737157 -->
# Fine-Tuning Guide for pi05_libero on Local Dataset

This guide explains how to fine-tune the pi05_libero checkpoint on your collected LIBERO rollout data.

## Overview

Your dataset is already collected at [`examples/libero/data/datasets/libero_rollouts/`](examples/libero/data/datasets/libero_rollouts/) containing:

- 20 episodes with 5,540 frames
- 10 different tasks
- Images (256x256x3), wrist images (256x256x3), state (8-dim), and actions (7-dim)

The fine-tuning workflow consists of three main steps:

1. **Create training configuration** pointing to your local dataset
2. **Compute normalization statistics** for the dataset
3. **Fine-tune the model** using the training script

## Step 1: Create Training Configuration

Add a new training config in [`src/openpi/training/config.py`](src/openpi/training/config.py) (around line 755, after the existing `pi05_libero` config):

```python
TrainConfig(
    name="pi05_libero_local",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="libero_rollouts",  # Matches your local dataset folder name
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,  # Match pi05_libero settings
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
    num_train_steps=30_000,
)
```

**Key configuration details:**

- `name`: Unique identifier for this training configuration
- `repo_id="libero_rollouts"`: Must match your dataset folder name
- `prompt_from_task=True`: Load task instructions from the LeRobot dataset's `task` field
- `weight_loader`: Initializes from the pi05_libero checkpoint
- `action_horizon=10`: Predicts 10 future action steps
- `discrete_state_input=False`: Uses continuous state representation

## Step 2: Set Environment Variable

Since your dataset is saved locally, you need to tell LeRobot where to find it:

```bash
export HF_LEROBOT_HOME=/home/ubuntu/home-jwei/soar/examples/libero/data/datasets
```

**Important:** This points to the **parent directory** containing `libero_rollouts/`, not the `libero_rollouts/` directory itself.

The dataset will be loaded from: `$HF_LEROBOT_HOME/libero_rollouts`

## Step 3: Compute Normalization Statistics

Before fine-tuning, you must compute normalization statistics for your dataset. This is required for proper data preprocessing during training.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero_local
```

This will:

1. Load your dataset from `$HF_LEROBOT_HOME/libero_rollouts`
2. Compute mean and standard deviation for states and actions
3. Save the statistics to `./assets/libero_rollouts/`

These normalization statistics are **required** for training and will be automatically loaded during fine-tuning.

## Step 4: Fine-Tune the Model

Now you can fine-tune the pi05_libero model on your collected data.

### Basic Fine-Tuning (JAX)

```bash
# Run training with JAX
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_local \
    --exp-name=my_libero_finetune \
    --overwrite
```

**Training parameters:**

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Allows JAX to use up to 90% of GPU memory (vs. default 75%)
- `--exp-name`: Name for this training run
- `--overwrite`: Overwrite existing checkpoints if rerunning

The checkpoint will be saved to `checkpoints/pi05_libero_local/my_libero_finetune/`.

### Alternative: PyTorch Training

If you prefer PyTorch over JAX:

```bash
uv run scripts/train_pytorch.py pi05_libero_local \
    --exp_name my_libero_finetune \
    --batch_size 256 \
    --num_train_steps 30000
```

### Low-Memory Fine-Tuning (LoRA)

For systems with limited GPU memory, you can use LoRA fine-tuning. Add this additional config to [`src/openpi/training/config.py`](src/openpi/training/config.py):

```python
TrainConfig(
    name="pi05_libero_local_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="libero_rollouts",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,
)
```

Then run:

```bash
uv run scripts/train.py pi05_libero_local_lora --exp-name=my_libero_lora_finetune
```

### Multi-GPU Training

For faster training with multiple GPUs (PyTorch only):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_libero_local \
    --exp_name my_libero_finetune
```

## Step 5: Monitor Training

Training progress will be logged to:

- **Console output**: Real-time training metrics
- **Weights & Biases dashboard**: If configured, detailed metrics and visualizations

Expected timeline:

- Training will run for 30,000 steps
- Checkpoints saved periodically
- Final checkpoint at `checkpoints/pi05_libero_local/my_libero_finetune/30000/`

## Using Your Fine-Tuned Model

After fine-tuning, you can use the trained model for inference:

### 1. Start the Policy Server

```bash
uv run scripts/serve_policy.py \
    --config pi05_libero_local \
    --checkpoint ./checkpoints/pi05_libero_local/my_libero_finetune/30000
```

### 2. Run Evaluation

```bash
python examples/libero/main.py \
    --host localhost \
    --port 8000 \
    --save_dataset False
```

## Data Requirements Checklist

For successful fine-tuning, ensure you have:

- [x] **Dataset**: Collected rollouts saved in LeRobot format
  - Location: `examples/libero/data/datasets/libero_rollouts/`
  - Contains: images, wrist_images, state, actions, task

- [ ] **Normalization Statistics**: Computed from your dataset
  - Location: `./assets/libero_rollouts/`
  - Contains: mean and std for state and actions
  - Generated by: `scripts/compute_norm_stats.py`

- [ ] **Training Config**: Configuration in [`src/openpi/training/config.py`](src/openpi/training/config.py)
  - Defines: model architecture, dataset location, hyperparameters
  - Points to: your local dataset and normalization stats

- [x] **Base Model Weights**: Pre-trained checkpoint to initialize from
  - Using: `gs://openpi-assets/checkpoints/pi05_libero/params`
  - Automatically downloaded during training

- [x] **Data Transforms**: Properly configured in `LiberoInputs` and `LiberoOutputs`
  - Location: [`src/openpi/policies/libero_policy.py`](src/openpi/policies/libero_policy.py)
  - Handles: image preprocessing, state/action formatting

## Troubleshooting

### Dataset Not Found

If you see "Dataset not found" errors:

1. Check that `HF_LEROBOT_HOME` is set correctly: `echo $HF_LEROBOT_HOME`
2. Verify the dataset exists at `$HF_LEROBOT_HOME/libero_rollouts/`
3. Ensure the `repo_id` in your config matches the dataset folder name

### Normalization Stats Missing

If you see "Normalization stats not found" errors:

1. Run `uv run scripts/compute_norm_stats.py --config-name pi05_libero_local`
2. Check that stats are saved to `./assets/libero_rollouts/`
3. Verify your config's `assets` parameter points to the correct location

### Out of Memory

If training runs out of GPU memory:

1. **Reduce batch_size**: Try 128, 64, or 32 instead of 256
2. **Use LoRA fine-tuning**: Only updates adapter layers, much lower memory usage
3. **Enable gradient checkpointing**: Automatically enabled in PyTorch training
4. **Use mixed precision training**: Automatically enabled with `pytorch_training_precision="bfloat16"`

### Small Dataset Warning

With only 20 episodes, you may experience:

- **Overfitting**: Model memorizes training data rather than generalizing
- **High variance**: Performance may vary significantly between runs

**Recommendations:**

- Monitor validation/test performance carefully
- Consider reducing `num_train_steps` to 5,000-10,000
- Collect more data if possible (aim for 50-100 episodes per task)
- Use data augmentation if available

## Summary: Complete Workflow

```bash
# 1. Set environment variable (must be done in every terminal session)
export HF_LEROBOT_HOME=/home/ubuntu/home-jwei/soar/examples/libero/data/datasets

# 2. Compute normalization stats (only once)
uv run scripts/compute_norm_stats.py --config-name pi05_libero_local

# 3. Fine-tune the model
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_local \
    --exp-name=my_libero_finetune \
    --overwrite

# 4. Serve the fine-tuned model
uv run scripts/serve_policy.py \
    --config pi05_libero_local \
    --checkpoint ./checkpoints/pi05_libero_local/my_libero_finetune/30000

# 5. Run evaluation (in another terminal)
python examples/libero/main.py \
    --host localhost \
    --port 8000 \
    --save_dataset False
```

### To-dos

- [ ] Add pi05_libero_local training config to config.py
- [ ] Export HF_LEROBOT_HOME environment variable
- [ ] Compute normalization statistics
- [ ] Execute fine-tuning with JAX
# Fine-Tuning Guide for LIBERO Dataset

This guide explains how to use the collected rollout data for fine-tuning Pi0 models.

## Overview

The fine-tuning workflow consists of three main steps:
1. **Collect rollout data** using `main.py`
2. **Compute normalization statistics** for the dataset
3. **Fine-tune the model** using the training script

## Step 1: Collect Rollout Data

The `main.py` script collects rollouts from the LIBERO environment and saves them in LeRobot format.

### Dataset Storage Location

By default, the dataset is saved to `examples/libero/data/libero_rollouts/`. This location is controlled by the `dataset_local_dir` parameter:

```bash
python examples/libero/main.py \
    --dataset_local_dir examples/libero/data \
    --dataset_repo_id libero_rollouts \
    --save_dataset True
```

The dataset will be saved at: `{dataset_local_dir}/{dataset_repo_id}/`

### Dataset Structure

The collected dataset includes:
- **Images**: Third-person camera view (256x256x3)
- **Wrist Images**: Wrist-mounted camera view (256x256x3)
- **State**: Proprioceptive state (8-dimensional: 3 for position, 4 for quaternion, 1 for gripper)
- **Actions**: Robot actions (7-dimensional: 3 for position, 3 for axis-angle rotation, 1 for gripper)
- **Task**: Language instruction for the task
- **Success**: Boolean flag indicating episode success

### Key Parameters

- `--save_dataset`: Whether to save rollouts (default: True)
- `--dataset_repo_id`: Name of the dataset (default: "libero_rollouts")
- `--dataset_local_dir`: Local directory to save dataset (default: "examples/libero/data")
- `--save_only_success`: If True, only save successful episodes (default: False)
- `--task_suite_name`: Which LIBERO task suite to use (default: "libero_spatial")
- `--num_trials_per_task`: Number of rollouts per task (default: 50)


### Dataset
    - Trajectory (Done)
    - Save all videos (failure/success) -> when saving
    - JSON file make up of
        - task_id: 0
            runs: {
                iteration: 0
                sccess: 0/1
                path_video: each video_file
                path_roll_out_data: each trajectory
            }
        So each task has a list of runs or dict of run or something with the iteration that we are on, and where it sucedded or not (0/1). And then the metadata (path_video, path_roll_out_data).
    


{
task_id: 0
runs: {
    iteration: 0
    sccess: 0/1
}


}
## Step 2: Compute Normalization Statistics

Before fine-tuning, you need to compute normalization statistics for your dataset. This is required for proper data preprocessing during training.

### Create a Training Config

First, you need to create a training configuration that points to your local dataset. Add this to `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi0_libero_local",
    model=pi0_config.Pi0Config(),
    data=LeRobotLiberoDataConfig(
        # Point to your local dataset
        repo_id="libero_rollouts",
        base_config=DataConfig(
            prompt_from_task=True,
        ),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```

### Set Environment Variable

Since your dataset is saved locally, you need to tell LeRobot where to find it:

```bash
export HF_LEROBOT_HOME=/absolute/path/to/examples/libero/data
```

For example:
```bash
export HF_LEROBOT_HOME=/home/ubuntu/home-phd/soar/examples/libero/data
```

### Compute Statistics

Run the normalization statistics computation script:

```bash
python scripts/compute_norm_stats.py pi0_libero_local
```

This will:
1. Load your dataset from `$HF_LEROBOT_HOME/libero_rollouts`
2. Compute mean and standard deviation for states and actions
3. Save the statistics to `./assets/pi0_libero_local/libero_rollouts/`

These normalization statistics are **required** for training and will be automatically loaded during fine-tuning.

## Step 3: Fine-Tune the Model

Now you can fine-tune the Pi0 model on your collected data.

### Basic Fine-Tuning

```bash
# Set the dataset location
export HF_LEROBOT_HOME=/absolute/path/to/examples/libero/data

# Run training
python scripts/train_pytorch.py pi0_libero_local \
    --exp_name my_libero_finetune \
    --batch_size 32 \
    --num_train_steps 30000
```

### Low-Memory Fine-Tuning (LoRA)

For systems with limited GPU memory, use LoRA fine-tuning:

```python
# Add this config to src/openpi/training/config.py
TrainConfig(
    name="pi0_libero_local_lora",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="libero_rollouts",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
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
python scripts/train_pytorch.py pi0_libero_local_lora --exp_name my_libero_lora_finetune
```

### Multi-GPU Training

For faster training with multiple GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi0_libero_local \
    --exp_name my_libero_finetune
```

## Data Requirements Checklist

For successful fine-tuning, ensure you have:

- [x] **Dataset**: Collected rollouts saved in LeRobot format
  - Location: `examples/libero/data/libero_rollouts/`
  - Contains: images, wrist_images, state, actions, task, success

- [x] **Normalization Statistics**: Computed from your dataset
  - Location: `./assets/pi0_libero_local/libero_rollouts/`
  - Contains: mean and std for state and actions
  - Generated by: `scripts/compute_norm_stats.py`

- [x] **Training Config**: Configuration in `src/openpi/training/config.py`
  - Defines: model architecture, dataset location, hyperparameters
  - Points to: your local dataset and normalization stats

- [x] **Base Model Weights**: Pre-trained checkpoint to initialize from
  - Default: `gs://openpi-assets/checkpoints/pi0_base/params`
  - Automatically downloaded during training

- [x] **Data Transforms**: Properly configured in `LiberoInputs` and `LiberoOutputs`
  - Location: `src/openpi/policies/libero_policy.py`
  - Handles: image preprocessing, state/action formatting

## Troubleshooting

### Dataset Not Found
If you see "Dataset not found" errors:
1. Check that `HF_LEROBOT_HOME` is set correctly
2. Verify the dataset exists at `$HF_LEROBOT_HOME/libero_rollouts/`
3. Ensure the `repo_id` in your config matches the dataset folder name

### Normalization Stats Missing
If you see "Normalization stats not found" errors:
1. Run `python scripts/compute_norm_stats.py pi0_libero_local`
2. Check that stats are saved to `./assets/pi0_libero_local/libero_rollouts/`
3. Verify your config's `assets` parameter points to the correct location

### Out of Memory
If training runs out of GPU memory:
1. Reduce `batch_size` (try 16 or 8)
2. Use LoRA fine-tuning instead of full fine-tuning
3. Enable gradient checkpointing (automatically enabled in PyTorch training)
4. Use mixed precision training (automatically enabled with `pytorch_training_precision="bfloat16"`)

## Advanced: Using Your Dataset for Inference

After fine-tuning, you can use the trained model for inference:

1. Start the policy server with your fine-tuned checkpoint:
```bash
python scripts/serve_policy.py \
    --config pi0_libero_local \
    --checkpoint ./checkpoints/pi0_libero_local/my_libero_finetune/30000
```

2. Run evaluation with the policy server:
```bash
python examples/libero/main.py \
    --host localhost \
    --port 8000 \
    --save_dataset False
```

## Summary

The complete workflow:

```bash
# 1. Collect data (saves to examples/libero/data/libero_rollouts/)
python examples/libero/main.py --save_dataset True

# 2. Set environment variable
export HF_LEROBOT_HOME=/absolute/path/to/examples/libero/data

# 3. Compute normalization stats
python scripts/compute_norm_stats.py pi0_libero_local

# 4. Fine-tune the model
python scripts/train_pytorch.py pi0_libero_local --exp_name my_finetune

# 5. Serve the fine-tuned model
python scripts/serve_policy.py \
    --config pi0_libero_local \
    --checkpoint ./checkpoints/pi0_libero_local/my_finetune/30000
```

All required data for fine-tuning is now properly configured and connected!

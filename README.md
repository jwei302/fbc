# FBC: Filtered Behavior Cloning for Self-Training Vision–Language–Action Policies

## Author
Phuc Duong, Kunwoo Min, Jeffrey Wei

CPSC 5800: Introduction to Computer Vision

Yale University, Department of Computer Science

## Overview
Vision-Language-Action (VLA) policies often require task- and domain-specific fine-tuning to achieve strong manipulation performance, yet many pipelines lack a simple mechanism for continual improvement from deployment-time interaction. We propose **Filtered Behavior Cloning** (FBC), a lightweight self-training recipe that executes a pretrained policy, filters its rollouts to retain only successful episodes, and fine-tunes on these self-generated demonstrations using parameter-efficient LoRA updates. Using the Pi0.5-LIBERO checkpoint and evaluating on LIBERO-90, FBC yields measurable gains in overall success rate and improves performance on a majority of non-trivial tasks under a constrained rollout budget. Our results suggest that success-filtered self-training is a practical and scalable primitive for refining large VLA policies, motivating future work that increases self refinement and adds safeguards to prevent over-specialization under repeated self-training.

## Setup

**Dependencies**

To install the relevant requirements, first install [uv](https://docs.astral.sh/uv/), to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/). We used Python 3.10.12.

Once installed, first set up a virtual environment:

```bash
uv venv
source venv/bin/activate
```

Clone the submodules:
```bash
git submodule update --init --recursive
```

Then, install the requirements:
```bash
uv pip install -r requirements.txt
```

**Install LIBERO Package**

The editable install doesn't work properly with `uv`, so use a `.pth` file workaround similar to below:

```bash
# Create path file to add LIBERO to Python path
echo "/path/to/fbc/third_party/libero" > .venv/lib/python3.11/site-packages/libero_path.pth
```

Or use the absolute path:
```bash
echo "/lambda/nfs/home-phd/fbc/third_party/libero" > .venv/lib/python3.11/site-packages/libero_path.pth
```

**Training Environment**

We used a Ubuntu 22.04 linux system to run our code and a NVIDIA GPU H100 to train our model.

## Relevant Files Modified

The repository is forked over from the [openpi](https://github.com/Physical-Intelligence/openpi) repository. We made the following additions

- [examples/libero/main.py](examples/libero/main.py) - Added the ability to save the rollouts metadata. Previously, only the outcome video was saved. We now save the `LeRobotDataset` metadata so the data can be reconstructed into a new dataset, which we later filter by success for fine-tuning. We also added the ability to output a log file that shows the results of each task, how many iterations were run, whether each iteration was successful, and the metadata for each episode.
- [examples/libero/postprocess.py](examples/libero/postprocess.py) - Added a postprocessing script to only include trials that were successful. Saved a new `LeRobotDataset` with those successful trials only.
- [src/openpi/training/config.py](src/openpi/training/config.py) - Added `pi05_libero_success_lora` for fine-tuning with LoRa utilizing our success only `LeRobotDataset`.

## Baseline Evaluation

To run the baseline evaluation on the pre-trained `pi05_libero` use the follow the intructions below.

1. Start the server to serve the pre-trained LIBERO pi05 policy.

```bash
python scripts/serve_policy.py --env LIBERO
```

2. Run the evaluation on a specific task suite (e.g,. libero_spatial, libero_10, libero_90). For our project, we used libero_90, and ran 10 trials per task. 

```bash
python examples/libero/main.py \
--args.task-suite-name libero_10 \ 
--args.num-trials-per-task 2 \
```

A full list of configurable arguments can be found in [examples/libero/main.py](examples/libero/main.py).

## Postprocessing

We postprocessed our data by filtering out the original dataset for success only trials during our baseline run. 

Running success-only filtering
```bash
python post_process.py \
  --args.rollouts_log rollouts_log \
  --args.input_dataset input_dataset \
  --args.output_dataset output_dataset
```

## Fine-tuning
For custom fine-tuning, ad a new training config in [`src/openpi/training/config.py`](src/openpi/training/config.py). (See examples in the files). 

For this project our configuration uses LoRa under the name `pi05_libero_success_lora`. Follow the instructions below on how to run fine-tune on a dataset after adding the configuration.

```bash
HF_LEROBOT_HOME=<path_to_dataset> \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero_success_lora \
    --exp-name=my_libero_finetune \
    --overwrite
```

**Note**: 
- We need to set `HF_LEROBOT_HOME` to a Hugging Face path containing the dataset for fine-tuning, or directly to a local path that contains our dataset. For example, a local path could be `/home/ubuntu/home-phd/soar/examples/libero/data/dataset`.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` specifies how much GPU memory the program is allowed to use.


## Model Checkpoint & Data

Our fine-tuned model checkpoint, total rollouts, filtered successful rollouts, and metadata-json files can be found [here]().
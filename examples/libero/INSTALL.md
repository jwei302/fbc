# LIBERO Installation Guide (Python 3.11+)

Quick guide for running LIBERO examples with Python 3.11+ instead of the recommended Python 3.8.

## Prerequisites

- Python 3.11+ environment with `uv` package manager
- Git submodules initialized

## Installation Steps

### 1. Initialize Git Submodules

```bash
git submodule update --init --recursive
```

### 2. Install Core Dependencies

```bash
cd /path/to/soar
source .venv/bin/activate

# Install robosuite (specific version required)
uv pip install robosuite==1.4.1

# Install LIBERO dependencies
uv pip install bddl==1.0.1 hydra-core easydict einops future cloudpickle gym
uv pip install imageio[ffmpeg] tqdm tyro PyYaml opencv-python matplotlib

# Install lerobot for dataset functionality
uv pip install lerobot

# Install openpi-client
uv pip install -e packages/openpi-client
```

### 3. Install LIBERO Package

The editable install doesn't work properly with `uv`, so we use a `.pth` file workaround:

```bash
# Create path file to add LIBERO to Python path
echo "/path/to/soar/third_party/libero" > .venv/lib/python3.11/site-packages/libero_path.pth
```

Or use the absolute path:
```bash
echo "/lambda/nfs/home-phd/soar/third_party/libero" > .venv/lib/python3.11/site-packages/libero_path.pth
```

### 4. Configure LIBERO (First Run Only)

```bash
# This will prompt for dataset path configuration (choose 'N' for defaults)
echo "N" | python -c "from libero.libero import benchmark"
```

## Running LIBERO

### Terminal 1: Start Model Server

```bash
cd /path/to/soar
uv run scripts/serve_policy.py --env LIBERO
```

### Terminal 2: Run Evaluation

```bash
cd /path/to/soar
source .venv/bin/activate

# Run with correct argument format (note: use --args. prefix and dashes)
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1
```

## Key Differences from Python 3.8 Setup

1. **No need for Python 3.8**: Works with Python 3.11+ by using newer versions of `numba` and `llvmlite`
2. **Specific robosuite version**: Must use `robosuite==1.4.1` (not 1.5.1)
3. **`.pth` file workaround**: Required because `uv`'s editable install doesn't handle nested package structure
4. **Argument format**: Use `--args.task-suite-name` (dashes) not `--task_suite_name` (underscores)

## Available Task Suites

- `libero_spatial` (max_steps: 220)
- `libero_object` (max_steps: 280)
- `libero_goal` (max_steps: 300)
- `libero_10` (max_steps: 520)
- `libero_90` (max_steps: 400)

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'libero'`
- **Solution**: Ensure the `.pth` file exists and points to the correct path
- **Verify**: `python -c "from libero.libero import benchmark; print('OK')"`

**Issue**: Can't connect to server
- **Solution**: Ensure model server is running in another terminal
- **Check**: Server should be listening on `0.0.0.0:8000`

**Issue**: `llvmlite==0.36.0` version conflict
- **Solution**: Don't use the old `requirements.txt` - it's for Python 3.8 only
- **Use**: The newer versions installed above work with Python 3.11+


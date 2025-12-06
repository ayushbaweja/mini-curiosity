# mini-curiosity

Implementation of A2C baseline for the Pathak et al. (2017) Intrinsic Curiosity Module paper, now wired to the FrozenLake-v1 environment along with a k-step curiosity modification. 

## Overview

This project implements and compares two reinforcement learning approaches on the FrozenLake-v1 environment:

- **Baseline A2C**: Standard Advantage Actor-Critic algorithm
- **A2C + ICM**: A2C enhanced with Intrinsic Curiosity Module for curiosity-driven exploration

### Key Features

- **Intrinsic Curiosity Module (ICM)**: Full implementation of the Pathak et al. (2017) ICM with forward and inverse dynamics models
- **k-step Curiosity**: Extended ICM supporting k-step future prediction for improved exploration
- **Visualization Tools**: TensorBoard integration for training metrics and video recording for policy visualization
- **Extensible Design**: Easy adaptation to other Gymnasium environments

## Quick Setup

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
git clone https://github.com/ayushbaweja/mini-curiosity.git
cd mini-curiosity
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

## Quick Experiment
```bash
#Train and test baseline A2C and Train A2C + ICM 

python -m curiosity_a2c.main --mode both --timesteps 300000 --test-episodes 50

```

## Train

```bash
#Train baseline A2c
python -m curiosity_a2c.main --mode baseline --timesteps 300000

#Train ICM
python -m curiosity_a2c.main --mode icm --timesteps 300000

#Custom ICM parameters
#icm-beta: Forward vs Inverse Loss Balance. If beta = 1, no Inverse Model (features filtering). 
#icm-eta: Intrinsic Reward Scale
#icm-lr: ICM Learning Rate
python -m curiosity_a2c.main \
    --mode icm \
    --timesteps 300000 \
    --icm-beta 0.2 \
    --icm-eta 0.01 \
    --icm-lr 0.001
```

## Test
```bash

# Test ICM model
python -m curiosity_a2c.main --mode test --model-type icm --test-episodes 20

# Test baseline model
python -m curiosity_a2c.main --mode test --model-type baseline --test-episodes 20

# Test both and compares statistics
python -m curiosity_a2c.main --mode compare --test-episodes 50
```

## View Results

```bash
# TensorBoard
tensorboard --logdir ./logs/

# Record videos
python -m curiosity_a2c.record_videos --mode both --episodes 1 5 10

# Record ICM only
python -m curiosity_a2c.record_videos --mode single --model-type icm --episodes 1 5 10

#Record baseline only
python -m curiosity_a2c.record_videos --mode single --model-type baseline --episodes 1 5 10


#Train and records
python -m curiosity_a2c.main \
    --mode both \
    --timesteps 100000 \
    --record-videos \
    --video-episodes 1 5 10
```

## Notebook

`./icm_notebook.ipynb`

Note: we recommend using previous commands for reproducing the experiments results. The notebook is mainly for theoretical convenience to understand the algorithm. 


## Project Structure

```
mini-curiosity/
├── README.md
├── icm_notebook.ipynb
├── logs
    ├── baseline/
    ├── icm/
├── output-videos/
├── models/
├── src
│   └── curiosity_a2c
│       ├── baseline_a2c.py
│       ├── compare.py
│       ├── icm_a2c.py
│       ├── icm_module.py
│       ├── __init__.py
│       ├── main.py
│       ├── record_videos.py
│       └── utils.py
        └── wrappers.py
```

## Adapting to a different Gymnasium task
```python
# src/curiosity_a2c/utils.py
from curiosity_a2c.envs import make_frozenlake_env


def make_env(render_mode=None):
    """Create and wrap the default FrozenLake environment."""
    return make_frozenlake_env(map_name="8x8", is_slippery=True, render_mode=render_mode)
```

To target another task (e.g. CartPole), replace `make_env` with the wrappers you need and update save paths so models stay organised:

```python
# src/curiosity_a2c/utils.py
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


def make_env(render_mode=None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return Monitor(env)

# src/curiosity_a2c/baseline_a2c.py
def train_baseline_a2c(..., save_path="models/baseline/a2c_cartpole_baseline"): ...

# src/curiosity_a2c/icm_a2c.py
def train_a2c_with_icm(..., save_path="models/icm/a2c_cartpole_icm"): ...

# src/curiosity_a2c/main.py
parser.add_argument('--baseline-path', default='models/baseline/a2c_cartpole_baseline_final', ... )
parser.add_argument('--icm-path', default='models/icm/a2c_cartpole_icm_final', ... )
```

## Requirements
All Python package requirements are listed in `pyproject.toml` and can be installed by following the [Quick Setup](#quick-setup) instructions.

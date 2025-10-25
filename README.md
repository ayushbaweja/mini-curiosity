# mini-curiosity

Implementation of A2C baseline for the Pathak et al. (2017) Intrinsic Curiosity Module paper, now wired to the FrozenLake-v1 environment.


## Request GPU on Pace

```bash

srun --account=<account_name> --job-name=my_a100_job --partition=gpu-a100 --gres=gpu:a100:1 --cpus-per-task=2 --mem=32G --time=01:00:00 --pty /bin/bash 

```
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
#Train and text baseline A2c and Train A2c + ICM 

python -m curiosity_a2c.main --mode both --timesteps 100000 --test-episodes 50


## Train

```bash
#Train baseline A2c
python -m curiosity_a2c.main --mode baseline --timesteps 100000

#Train ICM
python -m curiosity_a2c.main --mode icm --timesteps 100000

#Custom ICM parameters
#icm-beta: Forward vs Inverse Loss Balance. If beta = 1, no Inverse Model (features filtering). 
#icm-eta: Intrinsic Reward Scale
#icm-lr: ICM Learning Rate
python -m curiosity_a2c.main \
    --mode icm \
    --timesteps 100000 \
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

Note: we recommend using previous commands for reproducing the experiments results. The notebook is mainly for theoretical convenience to understand the main code algo and the maths. 
For experiments, it should be observed that the A2C+ICM version performs better than the baseline A2C only one.


## Project Structure

```
mini-curiosity/
├── README.md
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

- Python ≥3.9
- uv (for fast dependency management)

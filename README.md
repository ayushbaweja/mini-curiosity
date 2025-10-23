# mini-curiosity

Implementation of A2C baseline for the Pathak et al. (2017) Intrinsic Curiosity Module paper.


## Request GPU on Pace

```bash

srun --account=gts-ur2 --job-name=my_a100_job --partition=gpu-a100 --gres=gpu:a100:1 --constraint=A100-40GB --cpus-per-task=2 --mem=32G --time=01:00:00 --pty /bin/bash 

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

## Train

```bash
#Train baseline A2c and Train A2c + ICM 
python -m curiosity_a2c.main --mode both --timesteps 100000 --test-episodes 50

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
│       ├── __pycache__
│       │   ├── baseline_a2c.cpython-310.pyc
│       │   ├── compare.cpython-310.pyc
│       │   ├── icm_a2c.cpython-310.pyc
│       │   ├── icm_module.cpython-310.pyc
│       │   ├── __init__.cpython-310.pyc
│       │   ├── main.cpython-310.pyc
│       │   ├── record_videos.cpython-310.pyc
│       │   └── utils.cpython-310.pyc
│       ├── record_videos.py
│       └── utils.py
```

## To change to new environment 
```bash
#In src/curiosity_a2c/utils.py:

def make_env():
    """Create and wrap the environment"""
    env = gym.make('MountainCar-v0')  # ← Change this
    env = Monitor(env)
    return env

```

And optionally change the save paths used: 

```bash
#In src/curiosity_a2c/baseline_a2c.py:
def train_baseline_a2c(
    # ... parameters ...
    save_path="a2c_cartpole_baseline"  # ← Changed
):

#In src/curiosity_a2c/icm_a2c.py:
def train_a2c_with_icm(
    # ... parameters ...
    save_path="a2c_cartpole_icm"  # ← Changed
):

#In src/curiosity_a2c/main.py:
parser.add_argument(
    '--baseline-path',
    type=str,
    default='a2c_cartpole_baseline_final',  # ← Changed
    help='Path to baseline model'
)

parser.add_argument(
    '--icm-path',
    type=str,
    default='a2c_cartpole_icm_final',  # ← Changed
    help='Path to ICM model'
)
```

## Requirements

- Python ≥3.9
- uv (for fast dependency management)

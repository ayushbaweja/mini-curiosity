# mini-curiosity

Implementation of A2C baseline for the Pathak et al. (2017) Intrinsic Curiosity Module paper.

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
python src/baseline_a2c.py
```

## View Results

```bash
# TensorBoard
tensorboard --logdir ./logs/a2c_mountaincar/tensorboard/

# Record videos
python src/record_videos.py 1 5 10
```

## Project Structure

```
mini-curiosity/
├── src/
│   ├── baseline_a2c.py      # A2C training
│   └── record_videos.py     # Video generation
├── pyproject.toml           # Dependencies (uv uses this)
└── README.md
```

## Requirements

- Python ≥3.9
- uv (for fast dependency management)

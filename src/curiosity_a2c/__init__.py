"""
Curiosity-driven A2C implementation for sparse reward environments
"""
from .baseline_a2c import train_baseline_a2c
from .icm_a2c import train_a2c_with_icm
from .icm_module import ICMModule, ICMCallback
from .compare import compare_models
from .utils import make_env, test_model

__version__ = "0.1.0"

__all__ = [
    'train_baseline_a2c',
    'train_a2c_with_icm',
    'ICMModule',
    'ICMCallback',
    'compare_models',
    'make_env',
    'test_model',
]
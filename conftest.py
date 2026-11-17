"""Root-level pytest configuration.

This file is loaded by pytest before any test module or package conftest.
It sets TF/Flax suppression env vars and sys.modules guard so transformers
uses PyTorch-only mode, preventing NumPy 2.x / TF 1.x ABI conflicts in the tf-gpu environment.
"""

import os
import sys

# Suppress TF in transformers to prevent broken C-extension import under NumPy 2.x
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None


def pytest_load_initial_conftests(early_config, parser, args):
    """Runs before any test or plugin imports."""
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = None



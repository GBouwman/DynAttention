"""Model utility functions for the DynVision toolbox.

This module provides model-related utilities:
- Error handling and logging
- Model stability checking
- Weight analysis and validation
"""

import logging
import traceback
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Tuple, Optional, Dict
from pathlib import Path
from .config_utils import filter_kwargs

import numpy as np
import torch

logger = logging.getLogger(__name__)


def handle_errors(verbose: bool = False) -> Callable:
    """Decorator to handle errors consistently across functions.

    Wraps functions with try-except that catches exceptions, logs them,
    and optionally prints full traceback before re-raising as ValueError.

    Args:
        verbose: Whether to print full Python traceback on error

    Returns:
        Decorated function with error handling

    Example:
        @handle_errors(verbose=True)
        def my_function():
            # Function code here
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if verbose:
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise ValueError(f"Error in {func.__name__}: {str(e)}") from e

        return wrapper

    return decorator


def check_stability(x: torch.Tensor, stage: str = "") -> None:
    """Check numerical stability of activations.

    Args:
        x: Tensor to check for stability
        stage: Optional name of the stage being checked (for logging)
    """
    if torch.isnan(x).any():
        logger.error(f"NaN values detected in {stage} output")
        # raise ValueError("Numerical instability detected")
    if torch.isinf(x).any():
        logger.error(f"Inf values detected in {stage} output")
        # raise ValueError("Numerical instability detected")


def check_weights(model, message="", min=-2, max=2):
    """Check model weights for potential issues.

    Analyzes weight distributions and identifies problematic values:
    - Values outside specified range
    - NaN values
    - Extremely small/large values

    Args:
        model: PyTorch model to check
        message: Optional message to display with results
        min: Minimum acceptable weight value
        max: Maximum acceptable weight value

    Returns:
        Tuple of (weight_info, contain_nan):
        - weight_info: Dictionary mapping layer names to weight statistics
        - contain_nan: Boolean indicating if any NaN values were found
    """
    layer_names = model.state_dict().keys()
    layer_names = [name.rstrip(".weight") for name in layer_names]
    weight_info = {}
    contain_nan = False

    for layer in model.state_dict().keys():
        layer_name = layer.rstrip(".weight")
        weights = model.state_dict()[layer].numpy().flatten()
        n_weights = len(weights)
        min_value = np.nanmin(weights)
        max_value = np.nanmax(weights)
        norm = np.linalg.norm(weights)
        weight_info[layer_name] = SimpleNamespace(
            min=min_value, max=max_value, norm=norm
        )

        is_nan = ~np.isfinite(weights)
        is_large = np.abs(weights) > max
        is_small = np.abs(weights) < min
        is_bad = is_nan.any() or is_small.any() or is_large.any()

        if is_bad:
            print(message)
            print(f"Layer: {layer_name}")
            if is_large.any():
                n_large = np.sum(is_large.astype(int))
                print(f"\t{n_large}/{n_weights} large weights (>{max:2f})")
            if is_small.any():
                n_small = np.sum(is_small.astype(int))
                print(f"\t{n_small}/{n_weights} small weights (<{min:3f})")
            if is_nan.any():
                n_nans = np.sum(is_nan.astype(int))
                print(f"\t{n_nans}/{n_weights} NaN weights")
                contain_nan = True

    return weight_info, contain_nan


@handle_errors(verbose=False)
def load_model_and_weights(
    model_name: str,
    state_dict_path: Path,
    config: Any,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, int]:
    """Load the model and its weights.

    Args:
        model_name: Name of the model class
        state_dict_path: Path to the saved model weights
        config: Configuration object containing model parameters

    Returns:
        Tuple containing:
            - Loaded model instance
            - Number of classes

    Raises:
        ValueError: If model loading fails
    """
    # Lazy import to avoid circular dependency
    from dynvision import models

    state_dict = torch.load(state_dict_path, map_location=device)
    if not len(state_dict):
        raise ValueError(f"State dict is empty: {state_dict_path}")

    last_key = next(reversed(state_dict))
    n_classes = len(state_dict[last_key])

    model_class = getattr(models, model_name)
    model_args, _ = filter_kwargs(model_class, vars(config))
    model_args.update({"n_classes": n_classes})

    model = model_class(**model_args)

    if device is not None:
        model = model.to(device)

    model.load_state_dict(state_dict)

    return model

def load_weights_for_training(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    keep_time_constants: bool = True,
) -> Dict[str, list]:
    """Load weights into a freshly built model before training (init or transfer).

    - Builds the lazily created skip/feedback convs first, so their weights are loaded
      instead of being dropped as unexpected keys.
    - With ``keep_time_constants``, ``tau`` entries are not loaded: the dynamics come from
      this run's config (a tau=dt backbone must not switch a tau=2dt model back to alpha=1).

    Returns a dict with ``missing``, ``unexpected`` and ``kept_tau`` key lists.
    """
    if hasattr(model, "_initialize_connections") and (
        getattr(model, "skip", False) or getattr(model, "feedback", False)
    ):
        model._initialize_connections()

    own_state = model.state_dict()
    kept_tau = []
    if keep_time_constants:
        tau_keys = [
            k for k in state_dict
            if k.rsplit(".", 1)[-1].startswith("tau") and k in own_state
        ]
        kept_tau = [
            k for k in tau_keys
            if not torch.equal(state_dict[k].detach().cpu(), own_state[k].detach().cpu())
        ]
        state_dict = {k: v for k, v in state_dict.items() if k not in tau_keys}

    result = model.load_state_dict(state_dict, strict=False)
    return {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "kept_tau": kept_tau,
    }

def load_trained_model(
    model_path: str | Path,
    config_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Rebuild a trained DynVision model as it was trained and load its weights.

    Reads the resolved training config written next to the weights
    (``trained.pt.config.yaml`` by default), rebuilds the model with the same
    constructor arguments, builds the lazily created skip/feedback connections,
    then loads the weights. Works for plain state dicts (``*.pt``) and Lightning
    checkpoints (``*.ckpt``).

    Returns:
        The model in eval mode on ``device``, and the resolved config as a flat
        dict (e.g. ``cfg["data.resolution"]``, ``cfg["data.normalize"]``).
    """
    import yaml
    from dynvision import models  # lazy import avoids a circular dependency

    model_path = Path(model_path)
    config_path = Path(config_path) if config_path else model_path.parent / "trained.pt.config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_class = getattr(models, cfg["model.model_name"])
    model_kwargs = {
        key.split(".", 1)[1]: value
        for key, value in cfg.items()
        if key.startswith("model.") and value is not None
    }
    model_kwargs, _ = filter_kwargs(model_class, model_kwargs)
    model_kwargs.pop("target_dtype", None)  # load in float32; float16 breaks construction on CPU
    model = model_class(**model_kwargs).to(device)

    # Lazily created skip/feedback convs must exist before loading, else their weights are dropped
    if hasattr(model, "_initialize_connections"):
        model._initialize_connections()

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)  # .ckpt or plain .pt
    result = model.load_state_dict(state_dict)
    if result is not None and (result.missing_keys or result.unexpected_keys):
        logger.warning(
            f"Loading {model_path.name}: missing keys {result.missing_keys}, "
            f"unexpected keys {result.unexpected_keys}"
        )
    return model.eval(), cfg
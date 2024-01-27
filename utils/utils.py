"""
Some utility functions for the SE training script, which includes:
    is_list: check if an input is a list or not
    load_checkpoint: load a pytorch checkpoint model from a path
    prepare_empty_dir: if the input dir-path doesn't exist, make a new directory.

"""
import random
import importlib
from typing import Union

import json5
import torch
import torch.nn as nn
import numpy as np

from typing import Sequence

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(config_path):
    """
    Load a JSON configuration file.
    
    Args:
        config_path (pathlib.Path): Path to the configuration file.
        
    Returns:
        dict: Configuration data.
    """
    with open(config_path) as f:
        config = json5.load(f)
    return config


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def initialize_config(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg, config items as follow:
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])



def load_pretrained_model(
    pretrained_model_config: dict, 
    pretrained_model_checkpoint_path: str = None,
    prefix: str = None,
    device: Union[str, torch.device] = "cuda:0"
) -> nn.Module:
    """
    Load a pre-trained model from a checkpoint.

    Args:
        pretrained_model_config (dict): Configuration dictionary for initializing the model.
        pretrained_model_checkpoint_path (str, optional): Path to the pre-trained model checkpoint. 
            If None, initializes the model without loading weights. Default is None.
        prefix (str, optional): Prefix to be removed from the state dictionary keys. If None, no prefix is removed.
        device (str): Device to load the model onto. Default is "cpu".

    Returns:
        nn.Module: The pre-trained model.

    Raises:
        AssertionError: If the initialized object is not an instance of nn.Module.
    """
    pretrained_model = initialize_config(pretrained_model_config)
    assert isinstance(pretrained_model, nn.Module), "The initialized object is not an instance of nn.Module."

    if pretrained_model_checkpoint_path:
        pretrained_model_checkpoint = torch.load(pretrained_model_checkpoint_path, map_location=device)
        state_dict = pretrained_model_checkpoint.get("state_dict", pretrained_model_checkpoint)
        
        if prefix is not None:
            state_dict = {key.replace(prefix, '', 1) if key.startswith(prefix) else key: value 
                          for key, value in state_dict.items()}
        
        pretrained_model.load_state_dict(state_dict)

    return pretrained_model.to(device)
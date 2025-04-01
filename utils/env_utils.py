import os
import random
from typing import Any, Dict, List

import numpy as np
import torch


import hydra

from omegaconf import DictConfig, OmegaConf

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def set_seed(
    seed: int = 42, deterministic: bool = True, benchmark: bool = False
) -> None:
    """Manually set seeds, deterministic and benchmark modes.

    Included seeds:
        - random.seed
        - np.random.seed
        - torch.random.manual_seed
        - torch.cuda.manual_seed
        - torch.cuda.manual_seed_all

    Also, manually set up deterministic and benchmark modes.

    Args:
        seed (int): Seed. Default to 42.
        deterministic (bool): deterministic mode. Default to True.
        benchmark (bool): benchmark mode. Default to False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        #torch.cuda.deterministic = deterministic
        torch.cuda.benchmark = benchmark
        torch.use_deterministic_algorithms(True, warn_only=True)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): Callbacks config.

    Returns:
        List[Callback]: List with all instantiated callbacks.
    """

    callbacks: List = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): Loggers config.

    Returns:
        List[LightningLoggerBase]: List with all instantiated loggers.
    """

    logger: List = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

"""
Training script for contrastive learning experiments.
Based on Lightning + Hydra implementation.
"""
import os
import numpy as np
np.float_ = np.float64

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from utils.env_utils import instantiate_callbacks, instantiate_loggers
from utils.pylogger import get_pylogger
from utils.logging_utils import log_hyperparameters
from data.utils import load_normalize_from_file, create_omegaconf_from_json


# Configuration for Hydra
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",
    "config_name": "train.yaml",
}

log = get_pylogger(__name__)

# Set project root
os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

# Register resolvers for OmegaConf
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("load_normalize_from_file", load_normalize_from_file)
OmegaConf.register_new_resolver("create_omegaconf_from_json", create_omegaconf_from_json)


def train(cfg: DictConfig):
    """
    Main training function. Sets up the environment, initializes models, data, and trains.
    
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        
    Returns:
        dict: Dictionary with metrics from training.
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        log.info(f"Setting seed to {cfg.seed}")
        seed_everything(cfg.seed, workers=True)

    # Initialize transforms
    log.info("Initializing data transforms")
    train_transform = hydra.utils.instantiate(
        cfg.train_transform,
        _recursive_=True,
        _convert_="all"
    )
    val_transform = hydra.utils.instantiate(
        cfg.val_transform,
        _recursive_=True,
        _convert_="all"
    )

    # Initialize data module
    log.info(f"Initializing data module <{cfg.data.data_module._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.data_module, _recursive_=False,
    )
    datamodule = datamodule(train_transform=train_transform, val_transform=val_transform)

    # Model initialization - either load from checkpoint or create new
    if cfg.get("load_model_from_ckpt"):
        log.info(f"Loading model from checkpoint <{cfg.load_model_from_ckpt}>")
        model: LightningModule = hydra.utils.instantiate(
            cfg.module._target_
        ).load_from_checkpoint(cfg.load_model_from_ckpt)
        
        # Update model config with values from current run
        if hasattr(model, 'weighted_sampling_config'):
            model.weighted_sampling_config.start_epoch = cfg.module.weighted_sampling_config.start_epoch
            model.weighted_sampling_config.update_frequency = 1
    elif cfg.get("load_encoder_from_ckpt"):
        log.info(f"Loading encoder from checkpoint <{cfg.load_encoder_from_ckpt}>")
        model_class = hydra.utils.get_class(cfg.module._target_)
        model_old = model_class.load_from_checkpoint(cfg.load_encoder_from_ckpt)
        model: LightningModule = hydra.utils.instantiate(cfg.module, _recursive_=True)
        model.base_encoder = model_old.base_encoder
        model.projection_head = model_old.projection_head
    else:
        # Initialize new model
        log.info(f"Initializing model <{cfg.module._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.module, _recursive_=True)

    # Initialize callbacks
    log.info("Initializing callbacks")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    
    # Initialize loggers
    log.info("Initializing loggers")
    logger = instantiate_loggers(cfg.get("logger"))

    # Initialize trainer
    log.info(f"Initializing trainer")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # Log hyperparameters
    if logger:
        log.info("Logging hyperparameters")
        log_hyperparameters({
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        })
        
        if cfg.get("log_grads", False):
            logger[0].watch(model, log='all')

    # Start training
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    # Get training metrics
    train_metrics = trainer.callback_metrics

    # Test the model if requested
    if cfg.get("test"):
        log.info("Testing the model")
        
        # Test with last checkpoint
        log.info("Testing with last checkpoint")
        last_ckpt_path = trainer.checkpoint_callback.last_model_path
        trainer.test(model=model, datamodule=datamodule, ckpt_path=last_ckpt_path)
        
        # Test with best checkpoint if available
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            log.info(f"Testing with best checkpoint: {best_ckpt_path}")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt_path)
        else:
            log.warning("Best checkpoint not found! Using current weights for testing.")
            trainer.test(model=model, datamodule=datamodule)

        # Test on balanced dataset if requested
        if cfg.get("test_balanced", False):
            log.info("Testing on balanced dataset")
            datamodule.subsample_balanced_test = True
            datamodule.setup()
            ckpt_path = best_ckpt_path if best_ckpt_path else None
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Collect all metrics
    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    """
    Main entry point for the training script.
    
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        
    Returns:
        dict: Dictionary with metrics from training.
    """
    # Log configuration
    if cfg.get("print_config", True):
        log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    
    # Train the model
    metric_dict = train(cfg)
    
    return metric_dict


if __name__ == "__main__":
    main()

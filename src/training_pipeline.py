import os
from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """
    Pipeline for training a PyTorch model. Can additionally evaluate the model on a testset.
    Outputs the mae values of the training dataset to a file which path is specified by config.maes_path.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing the following keys:
            - "seed": (int, optional) Seed for random number generators.
            - "train": (bool, optional) Flag to indicate whether to train the model. Default is `True`.
            - "test": (bool, optional) Flag to indicate whether to test the model. Default is `False`.
            - "optimized_metric": (str, optional) Metric to be used for hyperparameter optimization.
            - "datamodule": (Dict[str, Any]) Configuration dictionary for the data module.
            - "model": (Dict[str, Any]) Configuration dictionary for the model.
            - "trainer": (Dict[str, Any]) Configuration dictionary for the trainer.
            - "callbacks": (Dict[str, Dict[str, Any]], optional) Configuration dictionaries for the callbacks.
            - "logger": (Dict[str, Dict[str, Any]], optional) Configuration dictionaries for the loggers.
            - "wrapper": (Dict[str, Any]) Configuration dictionary for the wrapper module.

    Returns
    -------
    score : float, optional
        Score of the `optimized_metric` for hyperparameter optimization. Returned only if `optimized_metric` is
        provided in the configuration.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        log.info("Starting testing!")
        print(ckpt_path)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


    log.info(f"Calculating mean absolute errors over training dataset <{config.model._target_}>")
    #print(ckpt_path)
    #model.load_from_checkpoint(ckpt_path)
    wrappermodel: LightningModule = hydra.utils.instantiate(config.wrapper, ckpt_path=trainer.checkpoint_callback.best_model_path)

    maes = torch.cat(trainer.predict(model=wrappermodel, datamodule=datamodule)).flatten()

    maes, _ = torch.sort(maes)
    torch.save(maes, config.maes_path)
    log.info(f"Saving maes to %s" %str(config.maes_path))

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return trainer.checkpoint_callback.best_model_path

import copy
import logging
import os
import tempfile
from typing import List, Optional
import re
import hydra
import torch
import shutil
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from copy import deepcopy
from src import utils

log = utils.get_logger(__name__)

def create_new_model_version(config: DictConfig) -> Optional[float]:
    """
    Create a new model version by running the preprocessing and training pipelines.

    This function creates a new model version by running the preprocessing and training pipelines on a given dataset, specified in the `config` argument.
    The pipelines save the results to a specified run path. If the run path already exists, the function prompts the user to confirm whether they want
    to rerun the pipelines for the same dataset. The pipeline saves a ready to use explain.yaml config. To use that config, refer to the explain_with_model_version pipeline.

    config file specified in configs/new_model_version.yaml
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing for the preprocessing a dataset and training a new model version on it. The configuration should include the following keys:
            - "dataset_path": (str) The file path of the dataset to be use. The file name should be in the format "YYYY-MM-DD_HH-MM-SS_log-data.csv".
            - "features_to_ignore": (list) A list of feature names to be ignored during preprocessing.

    Returns
    -------
    Optional[float]
        Returns `None` if the model version creation process completed successfully.
    """

    from src.preprocessing_pipeline import preprocess
    from src.training_pipeline import train

    dataset_name = config.dataset_path.split("/")[-1]
    #input dataset names are expected in this format
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_log-data\.csv$", dataset_name)

    #get dataset name/date to set as run name
    run_name = config.dataset_path.split("/")[-1][:-13]

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    model_versions_path = os.path.join(config.data_dir, "model_versions")

    if not os.path.exists(model_versions_path):
        os.makedirs(model_versions_path)

    run_path = os.path.join(model_versions_path, run_name)


    if os.path.exists(run_path):
        if input("Path exists, do you want to rerun for this dataset? [y/n] ") == "y":
            shutil.rmtree(run_path)
        else:
            logging.info("Finished!")

            return None

    #if provided with abs path keep as is otherwise add project data dir path
    if not os.path.isabs(config.dataset_path):
        config.dataset_path = os.path.join(config.data_dir, config.dataset_path)

    os.makedirs(run_path)

    base_confg = DictConfig({'original_work_dir': None,
                             'data_dir': None,
                             'print_config': None,
                             'ignore_warnings': None,
                             'name': None})

    for key in base_confg:
        base_confg[key] = config[key]
    print(config)
    preprocess_config = copy.deepcopy(base_confg)
    preprocess_config.preprocessor = config.preprocessor
    preprocess_config.preprocessor.dataset_path = config.dataset_path
    preprocess_config.preprocessor.feat_transformers_path = os.path.join(run_path, "feature_transformers")
    preprocess_config.preprocessor.output_full_path = os.path.join(run_path, (run_name+"_trainset.csv"))
    preprocess_config.preprocessor.create_feature_transformers = True
    preprocess_config.preprocessor.features_to_ignore = config.features_to_ignore
    print(preprocess_config)
    preprocess(preprocess_config)

    train_config = copy.deepcopy(base_confg)
    train_config.train = True
    train_config.test = True
    train_config.datamodule = config.datamodule
    train_config.datamodule.dataset_path = preprocess_config.preprocessor.output_full_path
    train_config.model = config.model
    train_config.wrapper = config.wrapper
    train_config.callbacks = config.callbacks
    train_config.logger = config.logger
    train_config.trainer = config.trainer
    train_config.preprocessor = preprocess_config
    for logger in train_config.logger:
        train_config.logger[logger].save_dir = os.path.join(run_path, "training_logs")
    train_config.callbacks.model_checkpoint.dirpath = run_path
    train_config.model.features_to_ignore = config.features_to_ignore
    train_config.maes_path = os.path.join(run_path, "maes.pt")

    if config.gpus == 0:
        train_config.trainer.accelerator = "cpu"
    else:
        if config.gpus == 1:
            train_config.trainer.gpus = 1
            train_config.trainer.accelerator = "gpu"
        else:
            train_config.trainer.gpus = config.gpus
            train_config.trainer.strategy = "ddp"
            train_config.trainer.sync_batchnorm = True

    best_ckpt_path = train(train_config)

    os.makedirs(os.path.join(run_path, "explained"))


    explain_config = copy.deepcopy(base_confg)
    explain_config.explainer = config.explainer
    explain_config.preprocess = True
    explain_config.preprocessor = preprocess_config.preprocessor
    explain_config.preprocessor.dataset_path = None
    explain_config.explainer.raw_dataset_path = None
    explain_config.explainer.dataset_path = None
    explain_config.explainer.model_ckpt_path = best_ckpt_path
    explain_config.explainer.maes_path = train_config.maes_path
    explain_config.explainer.output_path = None
    explain_config.explainer.notebook_path = None
    explain_config.name = "explaining with model version " + run_name


    with open(os.path.join(run_path, "explain.yaml"), "w") as f:
        OmegaConf.save(explain_config, f)


    return None

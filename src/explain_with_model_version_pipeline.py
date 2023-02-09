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

def explain_with_model_version(config: DictConfig) -> Optional[float]:
    """
     Run the explaining pipeline for a given dataset and model version.

     This function runs the explaining pipeline, including preprocessing and explanation generation, for a given dataset specified
     by config.dataset_to_explain_path and a model version specified by config.model_version. The pipeline saves the results to a specified run path.
     If the run path already exists, the function prompts the user to confirm whether they want to rerun the
     pipeline for the same dataset.

     config file specified in configs/explain_with_model_version.yaml

     Parameters
     ----------
    config : Dict[str, Any]
        Configuration dictionary containing for the explaining pipeline. The configuration should include the following keys:
            - "dataset_to_explain_path": (str) The file path of the dataset to be explained. The file name should be in the format "YYYY-MM-DD_HH-MM-SS_log-data.csv".
            - "model_version": (str ) The version of the model to be used for explaining the dataset. The version name should be in the format "YYYY-MM-DD_HH-MM-SS".

     Returns
     -------
     Optional[float]
         Returns `None` if the explaining pipeline completed successfully.

     Raises
     ------
     AttributeError
         If the model version path does not exist in the expected location.
     """
    from src.explaining_pipeline import explain

    dataset_name = config.dataset_to_explain_path.split("/")[-1]
    #input dataset names are expected in this format
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_log-data\.csv$", dataset_name)

    #get dataset name/date to set as run name
    run_name = config.dataset_to_explain_path.split("/")[-1][:-13]

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    #get model path
    model_version_path = os.path.join(config.data_dir, "model_versions", config.model_version)

    if not os.path.exists(model_version_path):
        raise AttributeError("Model version path doesn't exist in the expected location: " + model_version_path)

    #load explain config
    with open(os.path.join(model_version_path, "explain.yaml"), "r") as f:
        explain_config = OmegaConf.load(f)

    #run path for the explaining
    run_path = os.path.join(model_version_path, "explained", run_name)

    #if provided with abs path keep as is otherwise add project data dir path
    if not os.path.isabs(config.dataset_to_explain_path):
        config.dataset_to_explain_path = os.path.join(config.data_dir, config.dataset_to_explain_path)

    #dataset already explained?
    if os.path.exists(run_path):
        if input("Path exists, do you want to rerun for this dataset? [y/n] ") == "y":
            shutil.rmtree(run_path)
        else:
            logging.info("Finished!")

            return None

    #setting run specific configs
    shap_folder_path = os.path.join(run_path, "shap_folder")
    os.makedirs(shap_folder_path)

    explain_config.preprocessor.dataset_path = os.path.join(config.data_dir, config.dataset_to_explain_path)
    explain_config.preprocessor.output_full_path = os.path.join(run_path, run_name + "_preprocessed.csv")
    explain_config.explainer.raw_dataset_path = explain_config.preprocessor.dataset_path
    explain_config.explainer.dataset_path = explain_config.preprocessor.output_full_path
    explain_config.explainer.output_path = shap_folder_path
    explain_config.name = "explaining dataset version %s with model version %s" %(run_name, config.model_version)
    explain_config.explainer.notebook_path = run_path
    explain_config.explainer.outlier_threshold = config.outlier_threshold
    explain(explain_config)

    logging.info("Dataset explaining finished!")
    logging.info("Results saved to: %s" %run_path)
    return None

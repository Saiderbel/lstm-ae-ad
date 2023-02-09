import hydra
from omegaconf import DictConfig

from src import utils

log = utils.get_logger(__name__)


def preprocess(config: DictConfig) -> None:
    """
    Main function for preprocessing data.

    config file specified in configs/preprocess.yaml

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing the following keys:
            - "seed": (int, optional) Seed for random number generators.
            - "preprocessor": (Dict[str, Any]) Configuration dictionary for the preprocessor.
            - "extras": (Dict[str, Any], optional) Configuration dictionary for optional utilities.

    Returns
    -------
        None
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.preprocess.preprocessor import Preprocessor

    # Applies optional utilities
    utils.extras(config)

    log.info(f"Instantiating the preprocessor <{config.preprocessor._target_}>")
    preprocessor: Preprocessor = hydra.utils.instantiate(config.preprocessor)

    log.info(f"Preprocessing file")
    preprocessor.preprocess()
    log.info(f"All done!")


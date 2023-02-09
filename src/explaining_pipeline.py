import hydra
from omegaconf import DictConfig

from src import utils

log = utils.get_logger(__name__)


def explain(config: DictConfig) -> None:
    """
    Main function for using a model to predict anomalies in a dataset and also explaining the output with shap. The results are then saved
    to a jupyter notebook that runs out of the box.

    config file specified in configs/explain.yaml

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
    from src.explainable.explainer import Explainer
    from src.preprocess.preprocessor import Preprocessor

    # Applies optional utilities
    utils.extras(config)
    if config.preprocess:
        # Init preprocessor
        log.info(f"Instantiating the preprocessor <{config.preprocessor._target_}>")
        preprocessor: Preprocessor = hydra.utils.instantiate(config.preprocessor)

        preprocessor.preprocess()

    # Init explainer
    log.info(f"Instantiating the explainer <{config.explainer._target_}>")
    explainer: Explainer = hydra.utils.instantiate(config.explainer)

    log.info(f"Explaining!")
    explainer.explain()
    log.info(f"All done!")



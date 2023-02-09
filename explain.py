import dotenv
import hydra
from omegaconf import DictConfig

from src import utils
import warnings
warnings.filterwarnings("ignore", category=Warning, module="shap")
warnings.filterwarnings("ignore", category=Warning)
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="explain.yaml")
def main(config: DictConfig):
    """
    Main function for explaining a PyTorch model's predictions.

    config file specified in configs/explain.yaml
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing the following keys:
            - "seed": (int, optional) Seed for random number generators.
            - "preprocess": (bool, optional) Flag to indicate whether to preprocess the data. Default is `False`.
            - "preprocessor": (Dict[str, Any], optional) Configuration dictionary for the preprocessor.
            - "explainer": (Dict[str, Any]) Configuration dictionary for the explainer.
            - "extras": (Dict[str, Any], optional) Configuration dictionary for optional utilities.

    Returns
    -------
    None
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.explaining_pipeline import explain

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return explain(config)


if __name__ == "__main__":
    main()

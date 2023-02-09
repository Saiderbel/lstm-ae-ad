import dotenv
import hydra
from omegaconf import DictConfig

from src import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="preprocess.yaml")
def main(config: DictConfig):
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
    from src.preprocessing_pipeline import preprocess

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return preprocess(config)

if __name__ == "__main__":
    main()

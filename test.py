import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):
    """
    Main function for evaluating a PyTorch model.

    config file specified in configs/test.yaml

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing the following keys:
            - "seed": (int, optional) Seed for random number generators.
            - "ckpt_path": (str) Path to checkpoint file.
            - "datamodule": (Dict[str, Any]) Configuration dictionary for the data module.
            - "model": (Dict[str, Any]) Configuration dictionary for the model.
            - "trainer": (Dict[str, Any]) Configuration dictionary for the trainer.
            - "logger": (Dict[str, Dict[str, Any]], optional) Configuration dictionaries for the loggers.
            - "extras": (Dict[str, Any], optional) Configuration dictionary for optional utilities.

    Returns
    -------
    Any
        Return value of the `test` function.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()

import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    """
    Main function for training a PyTorch model.

    config file specified in configs/train.yaml

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
            - "extras": (Dict[str, Any], optional) Configuration dictionary for optional utilities.

    Returns
    -------
    Any
        Return value of the `train` function.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()

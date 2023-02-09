import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="new_model_version.yaml")
def main(config: DictConfig):
    """
    Main function for training a new model version on a specified dataset.

    config file specified in configs/new_model_version.yaml

    could be called from command line with this format::

        $ python new_model_version.py dataset_path="path/to/set" 'features_to_ignore=["feat_name", .. ]' gpus=<num_of_gpus>


    The number of gpus should be 0 if training on cpu,
    example::

        $ python new_model_version.py dataset_path="/home/2022-12-25_23-00-00_log-data.csv" 'features_to_ignore=["Bytes (custom)"]' gpus=0



    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing for the preprocessing a dataset and training a new model version on it. The configuration should include the following keys:
            - "dataset_path": (str) The file path of the dataset to be use. The file name should be in the format "YYYY-MM-DD_HH-MM-SS_log-data.csv".
            - "features_to_ignore": (list) A list of feature names to be ignored during preprocessing.

    Returns
    -------
    None:
        Returns `None` if the pipeline completed successfully.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.new_model_version_pipeline import create_new_model_version

    # Applies optional utilities
    utils.extras(config)


    # Train model
    return create_new_model_version(config)


if __name__ == "__main__":
    main()

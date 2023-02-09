import dotenv
import hydra
from omegaconf import DictConfig

from src import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="explain_with_model_version.yaml")
def main(config: DictConfig):
    """
    Run the explaining pipeline with a specified model version.

    This function runs the explaining pipeline, including preprocessing and explanation generation, for a given dataset and model version specified in `config`. The pipeline saves the results in the specified model version's path under `explained/`. If the path already exists, the user will be prompted to confirm whether they want to rerun the pipeline for the same dataset.

    config file specified in configs/new_model_version.yaml

    could be called from command line with this format::

        $ python explain_with_model_version.py dataset_to_explain_path="/path/to/set" model_version=<model_version> outlier_threshold=<outlier_threshold>

    example::

        $ python explain_with_model_version.py dataset_to_explain_path="/home/2022-12-25_23-00-00_log-data.csv" model_version=2022-12-24_23-00-00 outlier_threshold=0.97


    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing for the explaining pipeline. The configuration should include the following keys:
            - "dataset_to_explain_path": (str) The file path of the dataset to be explained. The file name should be in the format "YYYY-MM-DD_HH-MM-SS_log-data.csv".
            - "model_version": (str ) The version of the model to be used for explaining the dataset. The version name should be in the format "YYYY-MM-DD_HH-MM-SS".

    Returns
    -------
    None:
        Returns `None` if the explaining pipeline completed successfully.

    Raises
    ------
    AttributeError
        If the model version path does not exist in the expected location.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.explain_with_model_version_pipeline import explain_with_model_version

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return explain_with_model_version(config)


if __name__ == "__main__":
    main()

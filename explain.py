import dotenv
import hydra
from omegaconf import DictConfig

from src import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="explain.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.explainable.explainer import Explainer
    from src.preprocess.preprocessor import Preprocessor

    # Applies optional utilities
    utils.extras(config)

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


if __name__ == "__main__":
    main()

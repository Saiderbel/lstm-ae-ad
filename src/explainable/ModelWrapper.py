import torch
import pytorch_lightning as pl
import logging
from src.models.model import ADModel

class ModelWrapper(pl.LightningModule):
    """Wrapper class for PyTorch Lightning models.

    This class provides a simple way to instantiate and use a PyTorch Lightning
    model, either by providing a pre-trained model or by specifying a checkpoint
    path to load a model from.

    Parameters
    ----------
    ad_module : src.models.model.ADModel, optional
        Pre-trained PyTorch Lightning model. If provided, the `ckpt_path` argument will be ignored.
    ckpt_path : str, optional
        Path to a checkpoint file to load a PyTorch Lightning model from. If not provided, the model will be initialized using the default initialization method.
    """
    def __init__(self, ad_module=None, ckpt_path=None):
        super(ModelWrapper, self).__init__()
        if ad_module != None:
            self.ad_module = ad_module
            if ckpt_path != None:
                logging.warning("checkpoint path will be ignored since the model is provided!")
        else:
            self.ad_module = self.init_model(ckpt_path)

    def forward(self, x):
        outputs = self.ad_module(x)
        diff = torch.mean(torch.abs(self.flatten(outputs) - self.flatten(x)), axis=1)

        return diff.view((-1, 1))


    def flatten(self, X):
        """Flattens a 3D tensor into a 2D tensor.

        Given a 3D tensor `X` with shape (batch_size, sequence_length, feature_dim), this method returns a 2D tensor with shape (batch_size, feature_dim) by selecting the last element of the sequence for each batch.

        Parameters
        ----------
        X : torch.Tensor
            3D tensor to be flattened.

        Returns
        -------
        torch.Tensor
            2D tensor with shape (batch_size, feature_dim).
        """

        flattened_X = torch.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]
        return (flattened_X)

    def init_model(self, ckpt_path):
        return ADModel.load_from_checkpoint(ckpt_path)

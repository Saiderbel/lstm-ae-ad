"""
A PyTorch LightningModule for an autoencoder model, consisting of an encoder and a decoder.
"""
import pytorch_lightning as pl
import torch
from torch import nn
from src.utils.utils import get_num_features
from typing import Any
from src.models.components.encoder import Encoder
from src.models.components.decoder import Decoder


class ADModel(pl.LightningModule):
    """
    A PyTorch LightningModule for an autoencoder model, consisting of an encoder and a decoder.

    config file specified in configs/model/admodel.yaml

    """
    def __init__(self, **kwargs):
        """
        Initialize the ADModel instance.

        Parameters
        ----------
        kwargs : dict
            Hyperparameters for the model, such as:
                hidden_dim_1: The number of hidden units in the first LSTM layer.
                hidden_dim_2: The number of hidden units in the second LSTM layer.
                features_to_ignor: featrues to not consider
                seq_length: The length of the input sequences
        """
        super(ADModel, self).__init__()

        #this is when a new model is being created and not loaded from a ckpt
        if "features_to_ignore" in kwargs:
            num_features = get_num_features(kwargs["features_to_ignore"])
            kwargs["num_features"] = num_features
            del kwargs["features_to_ignore"]

        self.save_hyperparameters()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x : tensor
            A tensor of shape (batch_size, seq_length, num_features) representing the input data.

        Returns
        -------
        tensor
            A tensor of shape (batch_size, seq_length, num_features) representing the output of the model.
        """
        batch_size = x.shape[0]
        encoded = self.encoder(x, batch_size)
        reconstruction = self.decoder(encoded)
        return reconstruction

    def predict_step(self, batch, batch_idx):
        """
        Make a prediction on a single batch of data.

        Parameters
        ----------
        batch : tensor
            A batch of data to make predictions on.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        tensor
            The output of the model for the given batch.
        """
        outputs = self(batch)
        return outputs


    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler for the model.

        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.0, amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "monitor": "val/loss",
            "frequency": 1
        },
    }

    def on_train_start(self):
        """Called at the start of the training process."""
        pass

    def step(self, batch: Any):
        """
        Computes the loss for a single batch of data.

        Parameters:
            batch: A batch of data to compute the loss for.

        Returns:
            The loss for the given batch.
        """
        output = self(batch)
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(output, batch)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """
        Compute the loss for a single batch of data.

        Parameters
        ----------
        batch : tensor
            A batch of data to compute the loss for.

        Returns
        -------
        float
            The loss for the given batch.
        """
        loss = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}


    def validation_step(self, batch: Any, batch_idx: int):
        """
        Perform a single validation step.

        This method is called for each batch in the validation set. It should compute the loss on the batch and log any relevant metrics.

        Parameters
        ----------
        batch : tensor
            A batch of data to perform a training step on.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict
            A dictionary containing the loss for this batch.
        """

        loss= self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        """
        Perform a single test step.

        This method is called for each batch in the test set. It should compute the loss on the batch and log any relevant metrics.

        Parameters
        ----------
        batch : tensor
            A batch of data to perform a training step on.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict
            A dictionary containing the loss for this batch.
        """
        loss= self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}


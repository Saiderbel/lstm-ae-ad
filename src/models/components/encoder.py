import pytorch_lightning as pl
from torch import nn


class Encoder(pl.LightningModule):
    """
    A PyTorch LightningModule for an encoder architecture, consisting of two LSTM layers.

    Parameters
    ----------
    hidden_dim_1 : int
        The number of hidden units in the first LSTM layer.
    hidden_dim_2 : int
        The number of hidden units in the second LSTM layer.
    num_features : int
        The number of input features.
    seq_length : int
        The length of the input sequences.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Encoder instance.

        Parameters
        ----------
        hidden_dim_1 : int
            The number of hidden units in the first LSTM layer.
        hidden_dim_2 : int
            The number of hidden units in the second LSTM layer.
        num_features : int
            The number of input features.
        seq_length : int
            The length of the input sequences.
        """
        super(Encoder, self).__init__()
        self.save_hyperparameters()


        self.layer1 = nn.LSTM(
            input_size=self.hparams.num_features,
            hidden_size=self.hparams.hidden_dim_1,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        self.layer2 = nn.LSTM(
            input_size=self.hparams.hidden_dim_1,
            hidden_size=self.hparams.hidden_dim_2,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

    def forward(self, x, batch_size):
        """
        Perform a forward pass through the encoder.

        Parameters
        ----------
        x : tensor
            A tensor of shape (batch_size, seq_length, num_features) representing the input data.
        batch_size : int
            The size of the batch.

        Returns
        -------
        tensor
            A tensor of shape (batch_size, hidden_dim_2) representing the hidden state of the second LSTM layer for each element in the batch.
        """
        x = x.reshape((batch_size, self.hparams.seq_length, self.hparams.num_features))

        x, (_, _) = self.layer1(x)

        x, (hidden_n, _) = self.layer2(x)

        return hidden_n[-1,:,:]
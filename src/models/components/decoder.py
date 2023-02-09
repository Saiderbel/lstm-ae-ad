import pytorch_lightning as pl
from torch import nn

class Decoder(pl.LightningModule):
    """
    A PyTorch LightningModule for a decoder architecture, consisting of two LSTM layers and a linear output layer.

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
        Initialize the Decoder instance.

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
        super(Decoder, self).__init__()
        self.save_hyperparameters()

        #self.num_features, self.seq_length = num_features, seq_length
        #self.hidden_dim_1, self.hidden_dim_2 = hidden_dim_2, hidden_dim_1

        self.layer1 = nn.LSTM(
            input_size=self.hparams.hidden_dim_2,
            hidden_size=self.hparams.hidden_dim_2,
            num_layers=1,
            batch_first=True,
            dropout=0.2,
        )

        self.layer2 = nn.LSTM(
            input_size=self.hparams.hidden_dim_2,
            hidden_size=self.hparams.hidden_dim_1,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        self.output_layer = nn.Linear(self.hparams.hidden_dim_1, self.hparams.num_features)

    def forward(self, x):
        """
        Perform a forward pass through the decoder.

        Parameters
        ----------
        x : tensor
            A tensor of shape (batch_size, hidden_dim_1) representing the input data.

        Returns
        -------
        tensor
            A tensor of shape (batch_size, seq_length, num_features) representing the output of the decoder.
        """
        x = x.unsqueeze(1).repeat(1, self.hparams.seq_length, 1)

        x, (hidden_n, cell_n) = self.layer1(x)
        x, (hidden_n, cell_n) = self.layer2(x)
        x = x.reshape((-1, self.hparams.seq_length, self.hparams.hidden_dim_1))

        return self.output_layer(x)

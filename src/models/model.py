import pytorch_lightning as pl
import torch
from torch import nn
from argparse import ArgumentParser
from typing import Any, List


class Encoder(pl.LightningModule):
    # Encoder Architecture
    def __init__(self, hidden_dim_1, hidden_dim_2):
        super(Encoder, self).__init__()

        self.num_features, self.seq_length = 22, 32
        self.hidden_dim_1, self.hidden_dim_2 = hidden_dim_1, hidden_dim_2

        self.layer1 = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_dim_1,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        self.layer2 = nn.LSTM(
            input_size=self.hidden_dim_1,
            hidden_size=self.hidden_dim_2,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

    def forward(self, x, batch_size):
        x = x.reshape((batch_size, self.seq_length, self.num_features))

        x, (_, _) = self.layer1(x)

        x, (hidden_n, _) = self.layer2(x)

        return hidden_n[-1,:,:]

class Decoder(pl.LightningModule):
    #Decoder Architecture
    def __init__(self, hidden_dim_1, hidden_dim_2):
        super(Decoder, self).__init__()

        self.num_features, self.seq_length = 22, 32
        self.hidden_dim_1, self.hidden_dim_2 = hidden_dim_2, hidden_dim_1

        self.layer1 = nn.LSTM(
            input_size=self.hidden_dim_1,
            hidden_size=self.hidden_dim_1,
            num_layers=1,
            batch_first=True,
            dropout=0.2,
        )

        self.layer2 = nn.LSTM(
            input_size=self.hidden_dim_1,
            hidden_size=self.hidden_dim_2,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )

        self.output_layer = nn.Linear(self.hidden_dim_2, self.num_features)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)

        x, (hidden_n, cell_n) = self.layer1(x)
        x, (hidden_n, cell_n) = self.layer2(x)
        x = x.reshape((-1, self.seq_length, self.hidden_dim_2))

        return self.output_layer(x)

class ADModel(pl.LightningModule):
    def __init__(self, **kwargs):

        super().__init__()
        self.save_hyperparameters()

        # model's encoder and decoder
        self.encoder = Encoder(self.hparams.hidden_dim_1, self.hparams.hidden_dim_2)
        self.decoder = Decoder(self.hparams.hidden_dim_1, self.hparams.hidden_dim_2)

    def forward(self, x):
        batch_size = x.shape[0]
        encoded = self.encoder(x, batch_size)
        reconstruction = self.decoder(encoded)
        return reconstruction

    def predict_step(self, batch, batch_idx):
        outputs = self(batch)
        return outputs


    def configure_optimizers(self):
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
        pass

    def step(self, batch: Any):
        output = self(batch)
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(output, batch)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}


    def validation_step(self, batch: Any, batch_idx: int):
        loss= self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}


    def test_step(self, batch: Any, batch_idx: int):
        loss= self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

import pytorch_lightning as pl
from .anomaly_dataset import AnomalyDataset
from torch.utils.data import DataLoader
from typing import Optional
import pandas as pd

# the whole datamodule including train and validation loaders
class ADDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str = "path/to/dir", batch_size: int = 32, seq_len: int = 32):
        super().__init__()
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train = None
        self.valid = None


    def setup(self, stage: Optional[str] = None):
        if stage == "predict":
            chunks = []
            for chunk in pd.read_csv(self.dataset_path, chunksize=55555, sep="\t"):
                chunks.append(chunk)
                # a Pandas DataFrame to store the imported Data

            df = pd.concat(chunks)

            self.pred = AnomalyDataset(df.to_numpy(), seq_len=self.seq_len)

        else:
            chunks = []
            for chunk in pd.read_csv(self.dataset_path, chunksize=55555, sep="\t"):
                chunks.append(chunk)
                # a Pandas DataFrame to store the imported Data

            df = pd.concat(chunks)
            train_valid_split = int(len(df.index) * 0.85)
            X_train = df.iloc[0:train_valid_split].to_numpy()
            X_valid = df.iloc[train_valid_split:len(df.index)].to_numpy()

            self.train = AnomalyDataset(X_train, seq_len=self.seq_len)
            self.valid = AnomalyDataset(X_valid, seq_len=self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)

    def pred_dataloader(self):
        return DataLoader(self.pred, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)

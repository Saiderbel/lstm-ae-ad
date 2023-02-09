import pytorch_lightning as pl
from src.datamodules.anomaly_dataset import AnomalyDataset
from torch.utils.data import DataLoader
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split

# the whole datamodule including train and validation loaders
class ADDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for loading and processing time series data for anomaly detection.

    config file specified in configs/datamodule/addatamodule.yaml

    Parameters
    ----------
    dataset_path : str, optional
        The path to the directory containing the dataset. Default is "path/to/dir".
    batch_size : int, optional
        The batch size to use for the dataloaders. Default is 32.
    seq_len : int, optional
        The length of the sequences to use for detecting anomalies. Default is 32.

    Attributes
    ----------
    dataset_path : str
        The path to the directory containing the dataset.
    seq_len : int
        The length of the sequences to use for detecting anomalies.
    batch_size : int
        The batch size to use for the dataloaders.
    train : AnomalyDataset
        The training dataset.
    valid : AnomalyDataset
        The validation dataset.
    pred : AnomalyDataset
        The dataset to use for prediction.
    """
    def __init__(self, dataset_path: str = "path/to/dir", batch_size: int = 32, seq_len: int = 32):
        super().__init__()
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train = None
        self.valid = None


    def setup(self, stage: Optional[str] = None):
        """
        Load and process the dataset.

        Parameters
        ----------
        stage : str, optional
            The stage for which the dataset is being prepared. Can be one of "train", "val", or "predict". Default is None.
        """
        if stage == "predict":
            chunks = []
            for chunk in pd.read_csv(self.dataset_path, chunksize=55555):
                chunks.append(chunk)
                # a Pandas DataFrame to store the imported Data

            df = pd.concat(chunks)

            self.pred = AnomalyDataset(df.to_numpy(), seq_len=self.seq_len)

        else:
            chunks = []
            for chunk in pd.read_csv(self.dataset_path, chunksize=55555):
                chunks.append(chunk)
                # a Pandas DataFrame to store the imported Data

            df = pd.concat(chunks)
            train_valid_split = int(len(df.index) * 0.8)
            valid_test_split = int(len(df.index) * 0.9)
            X_train = df.iloc[0:train_valid_split].to_numpy()
            X_valid = df.iloc[train_valid_split:valid_test_split].to_numpy()
            X_test = df.iloc[valid_test_split:len(df)].to_numpy()

            self.train = AnomalyDataset(X_train, seq_len=self.seq_len)
            self.valid = AnomalyDataset(X_valid, seq_len=self.seq_len)
            self.test = AnomalyDataset(X_test, seq_len=self.seq_len)

    def train_dataloader(self):
        """Returns a PyTorch DataLoader object for the training set.

        The DataLoader is configured with the following options:
            - batch_size: the batch size is set to self.batch_size
            - shuffle: set to False to keep the data in order
            - drop_last: set to True to drop the last batch if it is smaller than the batch size
            - pin_memory: set to True to enable faster data transfer to the GPU (if available)

        Returns:
            A PyTorch DataLoader object for the training set.
        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        """
        Returns a PyTorch DataLoader object for the training set.

        The DataLoader is configured with the following options:
            - batch_size: the batch size is set to self.batch_size
            - shuffle: set to False to keep the data in order
            - drop_last: set to True to drop the last batch if it is smaller than the batch size
            - pin_memory: set to True to enable faster data transfer to the GPU (if available)

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader object for the training set.
        """
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        """
        Returns a PyTorch DataLoader object for the test set.

        The DataLoader is configured with the following options:
            - batch_size: the batch size is set to self.batch_size
            - shuffle: set to False to keep the data in order
            - drop_last: set to True to drop the last batch if it is smaller than the batch size
            - pin_memory: set to True to enable faster data transfer to the GPU (if available)

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader object for the test set.
        """
        return DataLoader(self.test, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)

    def predict_dataloader(self):
        """
        Returns a PyTorch DataLoader object for the prediction set.

        The DataLoader is configured with the following options:
            - batch_size: the batch size is set to self.batch_size
            - shuffle: set to False to keep the data in order
            - drop_last: set to True to drop the last batch if it is smaller than the batch size
            - pin_memory: set to True to enable faster data transfer to the GPU (if available)

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader object for the prediction set.
        """
        return DataLoader(self.pred, batch_size=self.batch_size, shuffle = False, drop_last=True, pin_memory=True)


if __name__ == '__main__':
    dataset_path = "/tmp/ad/data/test_pipeline_pro.csv"
    batch_size = 32
    seq_len = 32

    data_module = ADDataModule(dataset_path, batch_size, seq_len)
    data_module.setup()

    # train_dataloader example
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        print(batch)
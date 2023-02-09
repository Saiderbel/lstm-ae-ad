from torch.utils.data import Dataset
import torch

# Outputs inpout sequences
class AnomalyDataset(Dataset):
    """
    A PyTorch dataset for detecting anomalies in a time series.

    Parameters
    ----------
    X : numpy.ndarray
        The time series data to use for detecting anomalies.
    seq_len : int, optional
        The length of the sequences to use for detecting anomalies. Default is 1.

    Attributes
    ----------
    X : numpy.ndarray
        The time series data to use for detecting anomalies.
    seq_len : int
        The length of the sequences to use for detecting anomalies.
    """

    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        a = self.X.__len__() - (self.seq_len-1)
        return a

    def __getitem__(self, index):
        """
        Return a sequence of `self.seq_len` elements from the time series data at the given index.

        Parameters
        ----------
        index : int
            The index of the time series data to retrieve.

        Returns
        -------
        torch.Tensor
            A tensor of shape (`self.seq_len`,) containing the selected sequence of data.
        """
        return torch.from_numpy(self.X[index:index+self.seq_len]).float()

    def __data_shape__(self):
        """
        Return the shape of a single element in the dataset.

        Returns
        -------
        tuple
            The shape of a single element in the dataset.
        """
        return self.__getitem__(0).shape

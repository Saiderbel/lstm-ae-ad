from torch.utils.data import Dataset
import torch

# Outputs inpout sequences
class AnomalyDataset(Dataset):
    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        a = self.X.__len__() - (self.seq_len-1)
        return a

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index:index+self.seq_len]).float()

    def __data_shape__(self):
        return self.__getitem__(0).shape
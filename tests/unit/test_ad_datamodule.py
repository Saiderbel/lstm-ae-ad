import os

import pytest
import torch

from src.datamodules.data_module import ADDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    datamodule = ADDataModule(batch_size=batch_size)
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_pred

    datamodule.setup(stage="train")

    assert datamodule.data_train and datamodule.data_val


    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()


    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

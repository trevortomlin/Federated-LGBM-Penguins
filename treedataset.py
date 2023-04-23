'''
License: Apache
Original Authors: Flower
https://github.com/adap/flower
Modified by: Trevor Tomlin
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch.utils.data import Dataset
from typing import Dict
from flwr.common import NDArray
from typing import Dict, Tuple, Union
from torch.utils.data import DataLoader, Dataset, random_split

class TreeDataset(Dataset):
    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.labels = labels
        self.data = data
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> Dict[int, NDArray]:
        label = self.labels[idx]
        data = self.data[idx, :]
        sample = {0: data, 1: label}
        return sample
    
def do_fl_partitioning(trainset: Dataset, testset: Dataset, pool_size: int, batch_size: Union[int, str], val_ratio: float=0.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // pool_size
    lengths = [partition_size] * pool_size
    if sum(lengths) != len(trainset):
      lengths[-1] = len(trainset) - sum(lengths[0:-1])
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_ratio)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders = None
    testloader = get_dataloader(testset, "test", batch_size)
    return trainloaders, valloaders, testloader

def get_dataloader(dataset: Dataset, partition: str, batch_size: Union[int, str]) -> DataLoader:
    if batch_size == "whole": 
        batch_size = len(dataset)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train"))

def gen_datasets():
    df = pd.read_csv("data/penguins_size.csv")
    X = df.drop(columns=["species"])
    y = df["species"]

    encoders = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for col in ["sex", "island"]:
        encoders[col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[col] = encoders[col].fit_transform(X_train[[col]])
        X_test[col] = encoders[col].transform(X_test[[col]])

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    encoders["species"] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    y_train = encoders["species"].fit_transform(y_train.reshape(-1, 1))
    y_test= encoders["species"].transform(y_test.reshape(-1, 1))

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    X_train.flags.writeable=True
    y_train.flags.writeable=True
    X_test.flags.writeable=True
    y_test.flags.writeable=True

    print("Feature dimension of the dataset:", X_train.shape[1])
    print("Size of the trainset:", X_train.shape[0])
    print("Size of the testset:", X_test.shape[0])
    assert(X_train.shape[1] == X_test.shape[1])

    trainset = TreeDataset(np.array(X_train, copy=True), np.array(y_train, copy=True))
    testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))

    return trainset, testset
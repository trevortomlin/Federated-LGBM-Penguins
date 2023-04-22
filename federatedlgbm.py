import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import flwr as fl
from flwr.common.typing import Parameters
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from flwr.common import NDArray, NDArrays
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_svmlight_file
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics import Accuracy, MeanSquaredError
from tqdm import trange, tqdm
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split


df = pd.read_csv("data/penguins_size.csv")

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

#print(df.head())

# X = df.drop(columns=["species"])
# y = df["species"]

# for col in ["island", "sex"]:
#     X[col] = X[col].astype("category")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = lgb.LGBMClassifier(
#     num_leaves=2048,
#     n_estimators=64,
#     #learning_rate=0.05,
# )
# model.fit(X_train, y_train)

# print(f"Accuracy: {100 * accuracy_score(y_test, model.predict(X_test)):2f}")

# ax = lgb.plot_tree(model, tree_index=53, figsize=(15, 15), show_info=['split_gain'])
# plt.show()
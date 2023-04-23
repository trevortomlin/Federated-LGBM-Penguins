from typing import Any, Optional
import lightgbm as lgb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Union
from flwr.common import NDArray

from treedataset import TreeDataset, get_dataloader

def construct_tree(dataset: Dataset, label: NDArray, n_estimators: int) -> lgb.LGBMClassifier:
    """Construct a tree model using the LightGBM library."""
    # Initialize model
    model = lgb.LGBMClassifier(
        num_leaves=2048,
        n_estimators=n_estimators,
        #learning_rate=0.05,
    )
    # Train model
    model.fit(dataset, label)
    return model

def construct_tree_from_loader(dataset_loader: DataLoader, n_estimators: int) -> lgb.LGBMClassifier:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators)

def single_tree_prediction(tree: lgb.LGBMClassifier, n_tree: int, dataset: NDArray) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    # num_t = len(tree.booster_.model_to_string())
    # if n_tree > num_t:
    #     print(
    #         "The tree index to be extracted is larger than the total number of trees."
    #     )
    #     return None

    return tree.predict(  # type: ignore
        dataset#, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )

def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[
        Tuple[lgb.LGBMClassifier, int],
        List[Tuple[lgb.LGBMClassifier, int]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
                temp_trees[i], j, x_train
            )

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(
        np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(
        np.expand_dims(y_train32, axis=-1)  # type: ignore
    )

    #print("x_train_enc32.shape: ", x_train_enc32.shape)

    return x_train_enc32, y_train32

def tree_encoding_loader(dataloader: DataLoader, 
    batch_size: int,
    client_trees: Union[
        Tuple[lgb.LGBMClassifier, int], 
        List[Tuple[lgb.LGBMClassifier, int]]
    ],
    client_tree_num: int,
    client_num: int,
) -> DataLoader:
    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num)
    if encoding is None:
        return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels)
    return get_dataloader(tree_dataset, "tree", batch_size)
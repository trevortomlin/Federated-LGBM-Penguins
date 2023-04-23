import pandas as pd
from sklearn.model_selection import train_test_split
import flwr as fl
from typing import Dict
from flwr.common import NDArray, NDArrays
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from config import *
from client import FL_Client
from server import FL_Server, serverside_eval
from tree import construct_tree
from treedataset import TreeDataset, do_fl_partitioning
import functools
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig
from typing import Dict
from flwr.common import Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History

#print(df.head())

df = pd.read_csv("data/penguins_size.csv")
X = df.drop(columns=["species"])
y = df["species"]

from sklearn.preprocessing import OrdinalEncoder

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


print("Feature dimension of the dataset:", X_train.shape[1])
print("Size of the trainset:", X_train.shape[0])
print("Size of the testset:", X_test.shape[0])
assert(X_train.shape[1] == X_test.shape[1])

trainset = TreeDataset(np.array(X_train, copy=True), np.array(y_train, copy=True))
testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))

def print_model_layers(model: nn.Module) -> None:
    print(model)
    for param_tensor in model.state_dict():
      print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def start_experiment(
    trainset: Dataset,
    testset: Dataset,
    num_rounds: int = 5, 
    client_tree_num: int = 50,
    client_pool_size: int = 5, 
    num_iterations: int = 100, 
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    batch_size: int = 32,
    val_ratio: float = 0.1) -> History:
    client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    # Partition the dataset into subsets reserved for each client.
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set 
    # (good for testing how the final model performs on the client's local unseen data)
    trainloaders, valloaders, testloader = do_fl_partitioning(trainset, testset, batch_size='whole', pool_size=client_pool_size, val_ratio=val_ratio)
    print(f"Data partitioned across {client_pool_size} clients"
          f" and {val_ratio} of local dataset reserved for validation.")

    # Configure the strategy
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # FedXgbNnAvg
    strategy = FedXgbNnAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
        evaluate_fn=functools.partial(serverside_eval, testloader=testloader, \
            batch_size=batch_size, client_tree_num=client_tree_num, client_num=client_num),
        accept_failures=False,
    )

    print(f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool.")
    print(f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}.")

    def client_fn(cid: str) -> fl.client.Client:
        """Creates a federated learning client"""
        if val_ratio > 0.0 and val_ratio <= 1.0:
            return FL_Client(trainloaders[int(cid)], valloaders[int(cid)], \
                client_tree_num, client_pool_size, cid, log_progress=False)
        else:
            return FL_Client(trainloaders[int(cid)], None, \
                client_tree_num, client_pool_size, cid, log_progress=False)

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
        num_clients=client_pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds), strategy=strategy)
    
    print(history)

    return history

start_experiment(trainset=trainset, testset=testset, num_rounds=20, client_tree_num=client_tree_num, \
    client_pool_size=client_num, num_iterations=100, batch_size=64, fraction_fit=1.0, min_fit_clients=1, val_ratio=0.0)

# global_tree = construct_tree(X_train, y_train, client_tree_num)
# preds_train = global_tree.predict(X_train)
# preds_test = global_tree.predict(X_test)

# result_train = accuracy_score(y_train, preds_train)
# result_test = accuracy_score(y_test, preds_test)
# print("Global LGBM Training Accuracy: %f" % (result_train))
# print("Global LGBM Testing Accuracy: %f" % (result_test))

# client_trees_comparison = []
# trainloaders, _, testloader = do_fl_partitioning(trainset, testset, pool_size=client_num, batch_size="whole", val_ratio=0.0)

# for i, trainloader in enumerate(trainloaders):
#   for local_dataset in trainloader:
#     local_X_train, local_y_train = local_dataset[0], local_dataset[1]

#     #print(local_X_train.shape, local_y_train.shape)

#     tree = construct_tree(local_X_train, local_y_train, client_tree_num)
#     client_trees_comparison.append(tree)

#     preds_train = client_trees_comparison[-1].predict(local_X_train)
#     preds_test = client_trees_comparison[-1].predict(X_test)

#     result_train = accuracy_score(local_y_train, preds_train)
#     result_test = accuracy_score(y_test, preds_test)
#     print("Local Client %d XGBoost Training Accuracy: %f" % (i, result_train))
#     print("Local Client %d XGBoost Testing Accuracy: %f" % (i, result_test))
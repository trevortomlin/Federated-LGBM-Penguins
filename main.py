import flwr as fl
from typing import Dict
import torch.nn as nn
from torch.utils.data import Dataset
from config import *
from client import FL_Client
from server import FL_Server, serverside_eval
from treedataset import do_fl_partitioning, gen_datasets
import functools
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig
from typing import Dict
from flwr.common import Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History

def main():
    trainset, testset = gen_datasets()

    start_experiment(trainset=trainset, testset=testset, num_rounds=20, client_tree_num=client_tree_num, \
        client_pool_size=client_num, num_iterations=100, batch_size=64, fraction_fit=1.0, min_fit_clients=1, val_ratio=0.0)

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

def print_model_layers(model: nn.Module) -> None:
    print(model)
    for param_tensor in model.state_dict():
      print(param_tensor, "\t", model.state_dict()[param_tensor].size())

if __name__ == "__main__":
    main()
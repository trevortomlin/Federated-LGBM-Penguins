'''
License: Apache
Original Authors: Flower
https://github.com/adap/flower
Modified by: Trevor Tomlin
'''

import torch
import flwr as fl
from torch.utils.data import DataLoader
from typing import Tuple, Union, List
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters
from cnn import CNN
from flwr.common.typing import Parameters
from traintest import train, test
from config import *
import lightgbm as lgb

from tree import construct_tree_from_loader, tree_encoding_loader

class FL_Client(fl.client.Client):
    def __init__(self, trainloader: DataLoader, valloader: DataLoader, \
        client_tree_num: int, client_num: int, cid: str, log_progress: bool=False):
        """
        Creates a client for training `network.Net` on tabular dataset.
        """
        self.cid = cid
        self.tree = construct_tree_from_loader(trainloader, client_tree_num)
        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.trainloader = None
        self.valloader = None
        self.client_tree_num = client_tree_num
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress

        # instantiate model
        self.net = CNN()

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)


    def get_parameters(self, ins: GetParametersIns) -> Tuple[GetParametersRes, Tuple[lgb.LGBMClassifier, int]]:
        return [GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(self.net.get_weights())), (self.tree, int(self.cid))]


    def set_parameters(self, parameters: Tuple[Parameters, Union[Tuple[lgb.LGBMClassifier, int], List[Tuple[lgb.LGBMClassifier, int]]]]
    ) -> Union[Tuple[lgb.LGBMClassifier, int], List[Tuple[lgb.LGBMClassifier, int]]]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]


    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations = fit_params.config["num_iterations"]
        batch_size = fit_params.config["batch_size"]
        aggregated_trees = self.set_parameters(fit_params.parameters)
    
        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        self.trainloader = tree_encoding_loader(self.trainloader_original, batch_size, aggregated_trees, self.client_tree_num, self.client_num)
        self.valloader = tree_encoding_loader(self.valloader_original, batch_size, aggregated_trees, self.client_tree_num, self.client_num)

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_result, num_examples = \
            train(self.net, self.trainloader, device=self.device, 
                  num_iterations=num_iterations, log_progress=self.log_progress)
        print(f"Client {self.cid}: training round complete, {num_examples} examples processed")

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(fit_params.config), 
            num_examples=num_examples, 
            metrics={"loss": train_loss, "accuracy": train_result}
        ) 

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = test(self.net, self.valloader, device=self.device, log_progress=self.log_progress)

        # Return evaluation information
        print(f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, accuracy={result:.4f}")
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss, num_examples=num_examples, 
            metrics={"accuracy": result})
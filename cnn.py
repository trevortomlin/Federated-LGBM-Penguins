import torch, torch.nn as nn
from config import *
import numpy as np
from collections import OrderedDict
import flwr as fl

class CNN(nn.Module):
    def __init__(self, n_channel: int=64) -> None:
        super(CNN, self).__init__()
        n_out = 1
        self.conv1d = nn.Conv1d(1, n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0)
        self.layer_direct = nn.Linear(n_channel * client_num, n_out)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Identity = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1d(x))
        x = x.flatten(start_dim=1)
        x = self.ReLU(x)
        #x = self.Sigmoid(self.layer_direct(x))
        x = self.layer_direct(x)
        return x
  
    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [np.array(val.cpu().numpy(), copy=True) for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
          if v.ndim != 0:
            layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)
from typing import List, Tuple, Union
from flwr.common import EvaluateRes, FitRes
from flwr.server.client_proxy import ClientProxy

client_num = 5
client_tree_num = 500 // client_num

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
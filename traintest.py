from typing import Tuple
import torch
from torchmetrics import Accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import *
from cnn import CNN

def train(
    net: CNN,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool=True) -> Tuple[float, float, int]:

    # Define loss and optimizer
    #criterion = nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    pbar = tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f'TRAIN') if log_progress else iter(cycle(trainloader))

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):
        tree_outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(tree_outputs)
        #outputs = torch.argmax(outputs, dim=1).reshape(-1, 1).type(torch.float32)
        #outputs = torch.argmax(outputs, dim=1).reshape(-1, 1)

        #print(outputs, labels)

        loss = criterion(outputs, labels.squeeze().long())
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        n_samples += labels.size(0)


       
        # print(outputs, labels)
        # exit()

        #acc = Accuracy(task='multiclass', num_classes=3)(outputs, labels.type(torch.int))
        acc = Accuracy(task='multiclass', num_classes=3)(outputs.argmax(dim=1), labels.squeeze().type(torch.int)).item()
        total_result += acc * labels.size(0)

        if log_progress:
            pbar.set_postfix({
                "train_loss": total_loss/n_samples, 
                "train_acc": total_result/n_samples
            })
    if log_progress:
        print("\n")
    
    return total_loss/n_samples, total_result/n_samples, n_samples   
    

def test(
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool=True) -> Tuple[float, float, int]:

    """Evaluates the network on test data."""
    #criterion = nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)

            #print(tree_outputs.shape, labels.shape)
            #print(tree_outputs[0, 0, :])

            outputs = net(tree_outputs)
            #outputs = torch.argmax(outputs, dim=1).reshape(-1, 1)
            # Collected testing loss and accuracy statistics

            #print(f"Outputs: {torch.argmax(outputs, dim=1)}, Labels: {labels}")
            #exit()

            #print(outputs)

            #print(outputs.reshape(-1).shape, labels.shape)
            #print(torch.argmax(outputs, axis=1))

            # print(data[0].shape, data[1].shape)
            # print(tree_outputs.shape, labels.shape)
            # print(outputs.shape)
            # exit()

            #print(len(testloader))

            #.reshape(-1).long()

            #print(outputs.shape, labels.shape)

            #outputs = torch.argmax(outputs, dim=1).float()

            total_loss += criterion(outputs, labels.squeeze().long()).item()
            n_samples += labels.size(0)

            #outputs = torch.argmax(outputs, dim=1).reshape(-1, 1).type(torch.float32)

            #print(outputs, "AAAAA", labels)

            #print(outputs.cpu().argmax(dim=1).shape, labels.type(torch.int).cpu().shape)

            #acc = Accuracy(task='multiclass', num_classes=3)(outputs.cpu(), labels.type(torch.int).cpu())
            acc = Accuracy(task='multiclass', num_classes=3)(outputs.argmax(dim=1).cpu(), labels.type(torch.int).squeeze().cpu()).item()
            total_result += acc * labels.size(0)

            #print(total_loss, acc)

    if log_progress:
        print("\n")
    
    return total_loss/n_samples, total_result/n_samples, n_samples
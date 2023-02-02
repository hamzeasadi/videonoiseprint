import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer





dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_setp(net: nn.Module, data:DataLoader, opt:Optimizer, criterion:nn.Module):
    epochloss = 0.0
    numbatchs = len(data)
    net.train()
    for X in data:
        X = X.squeeze(dim=0)
        res = net(X)
        loss = criterion(res)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epochloss+=loss.item()

    return epochloss/numbatchs


def val_setp(net: nn.Module, data:DataLoader, opt:Optimizer, criterion:nn.Module):
    epochloss = 0.0
    numbatchs = len(data)
    net.eval()
    with torch.no_grad():
        for X in data:
            X = X.squeeze(dim=0)
            res = net(X)
            loss = criterion(res)
            epochloss+=loss.item()

    return epochloss/numbatchs




def main():
    pass




if __name__ == '__main__':
    main()
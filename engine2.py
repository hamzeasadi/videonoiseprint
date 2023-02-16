import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer





dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_setp(net: nn.Module, data:DataLoader, opt:Optimizer, criterion:nn.Module):
    epochloss = 0.0
    numbatchs = len(data)
    net.train()
    for X1, X2, Y in data:
        res1, res2 = net(X1.to(dev), X2.to(dev))
        loss = criterion(res1, res2, Y.to(dev))
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
        for X1, X2, Y in data:
            res1, res2 = net(X1.to(dev), X2.to(dev))
            loss = criterion(res1, res2, Y.to(dev))
            epochloss+=loss.item()
    return epochloss/numbatchs




def main():
    pass




if __name__ == '__main__':
    main()
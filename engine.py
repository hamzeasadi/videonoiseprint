import os
from typing import Optional

import torch
from torch import nn as nn
from torch.utils.data import DataLoader




def train_step(model:nn.Module, data_loader:DataLoader, opt, criterion, dev):
    epoch_loss = 0
    model.train()
    crt1 = nn.L1Loss()
    crt2 = nn.BCEWithLogitsLoss()
    num_batch = len(data_loader)

    for X, Y in data_loader:
        out1, out2, out3 = model(X.to(dev))
        y = torch.zeros_like(out2, requires_grad=False)
        loss1 = crt1(out2+out3, y) + crt2(out2, y)
        loss2 = criterion(out1, Y.to(dev))
        loss = loss1 + loss2
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    return epoch_loss/num_batch




if __name__ == '__main__':
    print(__file__)

    
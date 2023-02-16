import os, random, sys
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.nn import functional as F
import numpy as np


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_psd(x1, x2):
    x = torch.cat((x1, x2), dim=0).squeeze()
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r
    

class OneClassLoss(nn.Module):

    def __init__(self, reg) -> None:
        super().__init__()
        self.crt = nn.BCELoss()
        self.reg = reg


    def forward(self, R1, R2, Y):
        logits = torch.linalg.matrix_norm(torch.subtract(R1, R2), dim=(1,2))
        p = torch.softmax(-logits, dim=0)
        l1 = -self.reg*calc_psd(R1, R2)
        return self.crt(p, Y) + l1


def main():
    print(42)
    x1 = torch.randn(size=(10, 3, 3))
    x2 = torch.randn(size=(10, 3, 3))
    y = torch.randint(low=0, high=2, size=(10, ), dtype=torch.float32)
    ll = OneClassLoss(reg=0.1)
    l = ll(x1, x2, y)
    print(l)
    
 


if __name__ == '__main__':
    main()


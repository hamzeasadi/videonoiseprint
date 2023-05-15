import os

import torch
from torch import nn as nn
from torchvision import models
from torchinfo import summary
import lossfunction  

class ConstLayer(nn.Module):
    def __init__(self, ks, inch, outch, num_classes, dev):
        super().__init__()
        self.x1 = torch.zeros(size=(1, 1, ks, ks), device=dev)
        self.x1[:, :, ks//2, ks//2] = 1

        self.x2 = torch.ones(size=(1, 1, ks, ks), device=dev)
        self.x2[:, :, ks//2, ks//2] = 0

        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)
        # self.resnet.fc = nn.Identity()
        
        self.constlayer = nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=ks, stride=1, bias=False, padding='same')

        self.midconv = nn.Sequential(
            nn.Conv2d(in_channels=outch, out_channels=3, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(3), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2))

       

    def forward(self, x):
        out2 = self.constlayer(self.x1)
        out3 = self.constlayer(self.x2)
        x = self.constlayer(x)
        x = self.midconv(x)
        out1 = self.resnet(x)
        return  out1, out2, out3
    



if __name__ == '__main__':
    print(__file__)
    ks = 5
    inch = 1
    outch = 3
    x = torch.randn(1,1,450,450)
    net = ConstLayer(ks=ks, inch=inch, outch=outch, num_classes=28, dev='cpu')
    net.resnet.fc = nn.Identity()
    out1, out2, out3 = net(x)
    print(out1.shape)

    const = net.constlayer
    out1 = const(x)
    print(out1.shape)

 

   

    



    


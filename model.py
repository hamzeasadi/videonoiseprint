import os
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
import conf as cfg



class VideoPrint(nn.Module):

    def __init__(self, inch=1, depth: int=20) -> None:
        super().__init__()
        self.depth = depth
        self.inch = inch
        self.noisext = self.blks()

    def blks(self):
        firstlayer = nn.Sequential(nn.Conv2d(in_channels=self.inch, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.ReLU())
        lastlayer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding='same')

        midelayers = [firstlayer]
        for i in range(self.depth):
            layer=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5), nn.ReLU())
            midelayers.append(layer)
        
        midelayers.append(lastlayer)
        fullmodel = nn.Sequential(*midelayers)
        return fullmodel

    def forward(self, x):
        out = self.noisext(x)
        res = x[:, 0:1, :, :] - out
        return res




def main():
    x = torch.randn(size=(10, 3, 64, 64))
    model = VideoPrint(inch=3, depth=20)
    # summary(model, input_size=[10, 1, 64, 64])
    out = model(x)
    print(out.shape)



if __name__ == '__main__':
    main()
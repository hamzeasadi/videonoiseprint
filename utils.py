import os, random, sys
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.nn import functional as F
import numpy as np
import lossfunc2 as loss2



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, trainloss=0.1, valloss=0.1)

    def save_ckp(self, model: nn.Module, opt: Optimizer, epoch, fname: str, trainloss=0.1, valloss=0.1):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['trainloss'] = trainloss
        self.state['valloss'] = valloss
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state


def euclidean_distance_matrix(x):
    eps = 1e-8
    x = torch.flatten(x, start_dim=1)
    # dot_product = torch.mm(x, torch.transpose(x, 1, 0))
    # xt = x
    dot_product = torch.mm(x, x.t())
    squared_norm = torch.diag(dot_product)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    distance_matrix = F.relu(distance_matrix)
    # distformas = distance_matrix.clone()
    mask = (distance_matrix == 0.0).float()
    distance_matrix = distance_matrix.clone() + mask * eps
    distance_matrix = torch.sqrt(distance_matrix)
    distance_matrix = distance_matrix.clone()*(1.0 - mask)
    return distance_matrix


def calc_labels(batch_size, numcams):
    numframes = batch_size//numcams
    lbl_dim = numcams * numframes
    labels = torch.zeros(size=(lbl_dim, lbl_dim), device=dev, dtype=torch.float32)
    for i in range(0, lbl_dim, numframes):
        labels[i:i+numframes, i:i+numframes] = 1
    # for i in range(labels.size()[0]):
    #     labels[i,i]=0
    return labels


def calc_m(batch_size, numcams, m1, m2):
    lbls = calc_labels(batch_size=batch_size, numcams=numcams)
    for i in range(lbls.size()[0]):
        for j in range(lbls.size()[1]):
            if lbls[i, j] == 1:
                lbls[i, j] = m1
                
            elif lbls[i, j] == 0:
                lbls[i, j] = m2

    return lbls

def calc_psd(x):
    # x = x.squeeze()
    dft = torch.fft.fft2(x)
    avgpsd =  torch.mean(torch.mul(dft, dft.conj()).real, dim=0)
    r = torch.mean(torch.log(avgpsd)) - torch.log(torch.mean(avgpsd))
    return r
    

class OneClassLoss(nn.Module):
    """
    doc
    """
    def __init__(self, batch_size, num_cams, reg, m1, m2) -> None:
        super().__init__()
        self.bs = batch_size
        self.nc = num_cams
        self.reg = reg
        # self.m = calc_m(batch_size=batch_size, numcams=num_cams, m1=m1, m2=m2)
        # distmtxdim = num_cams * (batch_size//num_cams)
        # self.m = margin*torch.ones(size=(distmtxdim, distmtxdim), device=dev, dtype=torch.float32)
        # self.lbls = calc_labels(batch_size=batch_size, numcams=num_cams, m1=m1, m2=m2)
        # for i in range(self.lbls.size()[0]):
        #     self.lbls[i,i] = 0

        self.crt = nn.BCEWithLogitsLoss()
        self.newloss = loss2.SoftMLoss(batch_size=batch_size, framepercam=batch_size//num_cams, m1=m1, m2=m2)
        # self.crt = nn.BCELoss(reduction='mean')

    def forward(self, X):
        Xs = X.squeeze()

        # distmatrix = euclidean_distance_matrix(x=Xs)
        
        # for i in range(distmatrix.size()[0]):
        #     distmatrix[i,i] = 1e+10
        # newlogits = - torch.square(distmatrix)
        # logits = torch.softmax(newlogits, dim=1)
        
        # logitsmargin = logits + self.m
        # logits = self.m - torch.square(distmatrix)
        # l1 = self.crt(logits, self.lbls)
        l2 = self.reg*calc_psd(x=Xs)
        l3 = self.newloss(Xs)

        # # return l1+l3 - l2
        return l3 - l2

        # logits = self.m - torch.square(distmatrix)

        # return self.crt(logits, self.lbls) - l2



def main():
    print(42)
    epochs = 1000
    m1 = []
    m2 = []
    for i in range(epochs): 
        m1.append(int(max(1, 10000/(1+i))))
        m2.append(int(max(5, 100000/(1+i))))

    print(m1[300])
    print(m2[300])

    # print(calc_labels(batch_size=200, numcams=40))
    # print(calc_m(batch_size=200, numcams=40, m1=10, m2=100))


    
    
 


if __name__ == '__main__':
    main()


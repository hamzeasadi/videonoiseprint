
import os, random
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.nn import functional as F
import numpy as np



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
        self.m = calc_m(batch_size=batch_size, numcams=num_cams, m1=m1, m2=m2)
        # distmtxdim = num_cams * (batch_size//num_cams)
        # self.m = margin*torch.ones(size=(distmtxdim, distmtxdim), device=dev, dtype=torch.float32)
        self.lbls = calc_labels(batch_size=batch_size, numcams=num_cams)
        self.crt = nn.BCEWithLogitsLoss()

    def forward(self, X):
        Xs = X.squeeze()

        distmatrix = euclidean_distance_matrix(x=Xs)
        logits = self.m - torch.square(distmatrix)
        l1 = self.crt(logits, self.lbls)
        l2 = calc_psd(x=Xs)
        return l1 - self.reg*l2



def main():
    # lbls = calc_labels(batch_size=20, numcams=5)
    # print(lbls)
    # print(lbls.device, lbls.dtype)
    x1 = torch.randn(size=(9, 1, 64, 64))
    x2 = torch.randn(size=(9, 1, 64, 64))
    x = torch.cat((x1, x2), dim=0)
    xs = x.squeeze()
    # distmtx = euclidean_distance_matrix(x=xs)
    # print(distmtx)
    # print(torch.square(distmtx))
    # print(distmtx.shape)
    # loss = OneClassLoss(batch_size=20, num_cams=9, margin=3, reg=0.0001)
    # l = loss(x.squeeze())
    # print(l)
    # # distmtx = euclidean_distance_matrix(x.squeeze())
    # # print(distmtx)
    # x = torch.tensor([
    #     [0, 1, 10, 10], [2, 7, 1, 9]
    # ], dtype=torch.float32)
    # xs = 3*torch.softmax(x, dim=1)
    # z = 3 -xs
    # out = torch.sigmoid(z)
    # # xs = -torch.log_softmax(x, dim=1)
    # print(x)
    # print(xs)
    # print(z)
    # print(out)
    # # print(-torch.log(xs))

    M1 = 20000
    M2 = 600
    m1 = []
    m2 = []
    epochs = list(range(150))
    for epoch in epochs:
        y = int(max((M1//(1+2*epoch) -0.8*epoch), 18))
        x = max(M2//(1+1*epoch)+1, 6)
        m1.append(y)
        m2.append(x)

    print(m1)
    print(m2)
    print(sum(np.array(m1)<19))
    print(sum(np.array(m2)<7))


if __name__ == '__main__':
    main()



import os, random
import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.nn import functional as F


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
    distformas = distance_matrix.clone()
    mask = (distformas == 0.0).float()
    distance_matrix += mask * eps
    distance_matrix = torch.sqrt(distance_matrix)
    distance_matrix *= (1.0 - mask)
    return distance_matrix


def calc_labels(batch_size, numcams):
    numframes = batch_size//numcams
    lbl_dim = numcams * numframes
    labels = torch.zeros(size=(lbl_dim, lbl_dim), device=dev, dtype=torch.float32)
    for i in range(0, lbl_dim, numframes):
        labels[i:i+numframes, i:i+numframes] = 1
    return labels


class OneClassLoss(nn.Module):
    """
    doc
    """
    def __init__(self, batch_size, num_cams, margin) -> None:
        super().__init__()
        self.bs = batch_size
        self.nc = num_cams
        distmtxdim = num_cams * (batch_size//num_cams)
        self.m = margin*torch.ones(size=(distmtxdim, distmtxdim), device=dev, dtype=torch.float32)
        self.lbls = calc_labels(batch_size=batch_size, numcams=num_cams)
        self.crt = nn.BCEWithLogitsLoss()

    def forward(self, X):
        Xs = X.squeeze()
        distmatrix = euclidean_distance_matrix(x=Xs)
        logits = self.m - torch.square(distmatrix)
        return self.crt(logits, self.lbls)




def main():
    # lbls = calc_labels(batch_size=20, numcams=5)
    # print(lbls)
    # print(lbls.device, lbls.dtype)
    x1 = torch.ones(size=(9, 1, 3, 3))
    x2 = torch.randn(size=(9, 1, 3, 3))
    x = torch.cat((x1, x2), dim=0)
    # xs = x.squeeze()
    # distmtx = euclidean_distance_matrix(x=xs)
    # print(torch.square(distmtx))
    # print(distmtx.shape)
    loss = OneClassLoss(batch_size=20, num_cams=9, margin=100)
    l = loss(x.squeeze())
    print(l)


if __name__ == '__main__':
    main()


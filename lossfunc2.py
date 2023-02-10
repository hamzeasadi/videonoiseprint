import torch
import numpy as np
import utils
from torch import nn as nn
from math import comb


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def distmtxidxlbl(batch_size, frprcam):
    indexs = torch.tensor(list(range(batch_size)))
    idxandlbl = dict()
    for blk in range(0, batch_size, frprcam):
        for row in range(blk, blk+frprcam):
            rowidx = []
            rowlbl = []
            for i in range(row+1, blk+frprcam):
                idx = torch.cat(( indexs[:blk], indexs[i:i+1], indexs[blk+frprcam:]), dim=0)
                rowidx.append(idx)
                rowlbl.append(blk)

            if len(rowidx)>0:
                idxandlbl[row] = (rowidx, rowlbl)

    return idxandlbl







class SoftMLoss(nn.Module):
    def __init__(self, batch_size, framepercam) -> None:
        super().__init__()
        self.distlbl = distmtxidxlbl(batch_size=batch_size, frprcam=framepercam)
        self.crtsft = nn.CrossEntropyLoss()
        self.logitsize = batch_size - framepercam + 1



    def forward(self, x):
        xs = x.squeeze()
        distmtx = utils.euclidean_distance_matrix(xs)
        logits = torch.zeros(size=(self.logitsize, ))
        labels = []
        for rowidx, logitlblidx in self.distlbl.items():
            row = distmtx[rowidx]
            rowlogitsidx, rowlbls = logitlblidx
            for logitidx, logitlbl in zip(rowlogitsidx, rowlbls):
                logits = torch.vstack((logits, row[logitidx]))
                labels.append(logitlbl)
        finallogits = logits[1:]
        finallabels = torch.tensor(labels, device=dev, dtype=torch.long)
        return self.crtsft(finallogits, finallabels)







def main():
    batch_size = 200
    stp = 5
    x = torch.randn(size=(batch_size, 1, 64, 64))
    myloss = SoftMLoss(batch_size=batch_size, framepercam=stp)
    loss = myloss(x)
    print(loss)


if __name__ == '__main__':
    main()
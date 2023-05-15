import os
from typing import Optional

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

import utils
from conf import Paths
import model as m

paths = Paths()


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
        # print(f'constloass={loss1.item()}, clsloss={loss.item()}')

    return epoch_loss/num_batch


# def eval_step(ckpoint_num, path, max_pair, dev):
#     pos_pair, neg_pair = utils.pairs(path=path, max_pair=max_pair)
#     predictions = []
#     ground_truth = []
    
#     model = m.ConstLayer(ks=5, inch=1, outch=3, num_classes=28)
#     ckpoint = torch.load(os.path.join(paths.model, f'ckpoint_{ckpoint_num}.pt'), map_location=dev)
#     model.load_state_dict(ckpoint)

#     model.eval()

#     const = model.constlayer
#     model.resnet.fc = nn.Identity()
    
#     for cam_name, files in pos_pair.items():
#         for pair_file in files:
#             smpl1_path = os.path.join(path, cam_name, pair_file[0])
#             smpl2_path = os.path.join(path, cam_name, pair_file[1])
#             # smpl1 = 
        
    


if __name__ == '__main__':
    print(__file__)

    

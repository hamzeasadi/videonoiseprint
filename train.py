import os
import argparse

import torch 
from torch import nn

import model as m
import dataset as dst
import lossfunction as lf
import conf as cfg
import engine


paths = cfg.Paths()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    epochs = 100
    lr = 1e-3
    batch_size = 128
    num_cls = 28

    model = m.ConstLayer(ks=5, inch=1, outch=3, num_classes=num_cls)
    criterion = lf.AdaMagfaceLoss(NumClasses=28, InputFeatures=512, dev=dev)
    opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    loader = dst.create_train_loader(batch_size=batch_size)

    for epoch in range(epochs):
        epoch_loss = engine.train_step(model=model, data_loader=loader, opt=opt, criterion=criterion, dev=dev)
        torch.save(model.state_dict(), os.path.join(paths.model, f'ckpoint_{epoch}.pt'))
        print(f'{epoch} {epoch_loss}')


import os
import conf as cfg
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import optim
import utils
import datasetup as dst
import model as m
import engine
import argparse
import numpy as np


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--coord', action=argparse.BooleanOptionalAction)
parser.add_argument('--adaptive', action=argparse.BooleanOptionalAction)
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epochs', '-e', type=int, required=False, metavar='epochs', default=1)
parser.add_argument('--batch_size', '-bs', type=int, required=True, metavar='numbatches', default=198)
parser.add_argument('--margin1', '-m1', type=float, required=True, metavar='margin1', default=250)
parser.add_argument('--margin2', '-m2', type=float, required=True, metavar='margin2', default=100000)
parser.add_argument('--reg', '-r', type=float, required=True, metavar='reg', default=1.1)
parser.add_argument('--depth', '-dp', type=int, required=True, metavar='depth', default=15)


args = parser.parse_args()

def epochtom(epoch, M1, M2, adaptive=False):
    if adaptive:
        m1 = max(5, int(M1*np.exp(-epoch/5)))
        m2 = max(10, int(M2*np.exp(-epoch/10)))
        return m1, m2
    else:
        return 5, 10
    



def train(Net:nn.Module, optfunc:Optimizer, epochs, modelname, batch_size=200, coordaware=False):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    # traindata, valdata = dst.createdl()
    for epoch in range(epochs):
        m1, m2 = epochtom(epoch=epoch, M1=args.margin1, M2=args.margin2, adaptive=args.adaptive)
        lossfunctr = utils.OneClassLoss(batch_size=args.batch_size, num_cams=40, reg=args.reg, m1=m1, m2=m2)
        lossfuncvl = utils.OneClassLoss(batch_size=args.batch_size, num_cams=5, reg=args.reg, m1=m1, m2=m2)

        traindata, valdata = dst.create_loader(batch_size=batch_size, caware=coordaware)
        trainloss = engine.train_setp(net=Net, data=traindata, opt=optfunc, criterion=lossfunctr)
        valloss = engine.val_setp(net=Net, data=valdata, opt=optfunc, criterion=lossfuncvl)
        fname = f'{modelname}_{epoch}.pt'
        # if epoch%2 == 0:
        kt.save_ckp(model=Net, opt=optfunc, epoch=epoch, trainloss=trainloss, valloss=valloss, fname=fname)

        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")










def main():
    inch=1
    if args.coord:
        inch=3
    model = m.VideoPrint(inch=inch, depth=args.depth)
    model = nn.DataParallel(model)
    model.to(dev)
    optimizer = optim.Adam(params=model.parameters(), lr=3e-4)
    # crt = utils.OneClassLoss(batch_size=args.batch_size, num_cams=9, margin=args.margin, reg=args.reg)

    if args.train:
        train(Net=model, optfunc=optimizer, epochs=args.epochs, modelname=args.modelname, batch_size=args.batch_size, coordaware=args.coord)


    print(args)




if __name__ == '__main__':
    main()
import os
import conf as cfg
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import optim
import utils
import dataset2 as dst
import model2 as m
import engine2
import argparse
import numpy as np
import lossfunc3


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
        m1 = int(max(3, M1/(1+epoch)))
        m2 = int(max(5, M2/(1+epoch)))
        return m1, m2
    else:
        return M1, M2
    



def train(Net:nn.Module, optfunc:Optimizer, epochs, modelname, batch_size=200, coordaware=False):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    # traindata, valdata = dst.createdl()
    for epoch in range(epochs):
        # reg = 10 - epoch%10
        m1, m2 = epochtom(epoch=epoch, M1=args.margin1, M2=args.margin2, adaptive=args.adaptive)
        lossfunctr = lossfunc3.OneClassLoss(reg=args.reg)
        lossfuncvl = lossfunc3.OneClassLoss(reg=args.reg)

        traindata, valdata = dst.create_loader(batch_size=batch_size, caware=coordaware)
        trainloss = engine2.train_setp(net=Net, data=traindata, opt=optfunc, criterion=lossfunctr)
        valloss = engine2.val_setp(net=Net, data=valdata, opt=optfunc, criterion=lossfuncvl)
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
    optimizer = optim.Adam(params=model.parameters())
    # crt = utils.OneClassLoss(batch_size=args.batch_size, num_cams=9, margin=args.margin, reg=args.reg)

    if args.train:
        train(Net=model, optfunc=optimizer, epochs=args.epochs, modelname=args.modelname, batch_size=args.batch_size, coordaware=args.coord)


    print(args)




if __name__ == '__main__':
    main()
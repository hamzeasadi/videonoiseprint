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


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
parser.add_argument('--coord', action=argparse.BooleanOptionalAction)
parser.add_argument('--modelname', '-mn', type=str, required=True, default='None')
parser.add_argument('--epochs', '-e', type=int, required=False, metavar='epochs', default=1)
parser.add_argument('--batch_size', '-bs', type=int, required=True, metavar='numbatches', default=198)
parser.add_argument('--margin', '-m', type=float, required=True, metavar='margin', default=1.1)
parser.add_argument('--reg', '-r', type=float, required=True, metavar='reg', default=1.1)
parser.add_argument('--depth', '-dp', type=int, required=True, metavar='depth', default=15)

args = parser.parse_args()

def train(Net:nn.Module, optfunc:Optimizer, lossfunc:nn.Module, epochs, modelname, batch_size=198, coordaware=False):
    kt = utils.KeepTrack(path=cfg.paths['model'])
    # traindata, valdata = dst.createdl()
    for epoch in range(epochs):
        traindata, valdata = dst.create_loader(batch_size=batch_size, caware=coordaware)
        trainloss = engine.train_setp(net=Net, data=traindata, opt=optfunc, criterion=lossfunc)
        valloss = engine.val_setp(net=Net, data=valdata, opt=optfunc, criterion=lossfunc)
        fname = f'{modelname}_{epoch}.pt'
        if epoch%2 == 0:
            kt.save_ckp(model=Net, opt=optfunc, epoch=epoch, trainloss=trainloss, valloss=valloss, fname=fname)

        print(f"epoch={epoch}, trainloss={trainloss}, valloss={valloss}")










def main():
    inch=1
    if args.coord:
        inch=3
    model = m.VideoPrint(inch=inch, depth=15)
    # model = nn.DataParallel(model)
    model.to(dev)
    optimizer = optim.Adam(params=model.parameters(), lr=3e-4)
    crt = utils.OneClassLoss(batch_size=args.batch_size, num_cams=9, margin=args.margin, reg=args.reg)

    if args.train:
        train(Net=model, optfunc=optimizer, lossfunc=crt, epochs=args.epochs, modelname=args.modelname, batch_size=args.batch_size, coordaware=args.coord)


    print(args)




if __name__ == '__main__':
    main()
import conf as cfg
import torch
from torch.utils.data import Dataset, DataLoader
import os, random
import cv2
import numpy as np
from patchify import patchify




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def coordinate(High, Width):
    xcoord = torch.ones(size=(High, Width), dtype=torch.float32)
    ycoord = torch.ones(size=(High, Width), dtype=torch.float32)
    for i in range(High):
        xcoord[i, :] = 2*(i/High) - 1
    for j in range(Width):
        ycoord[:, j] = 2*(j/Width) - 1
    
    coord = torch.cat((xcoord.unsqueeze(dim=0), ycoord.unsqueeze(dim=0)), dim=0)
    return coord

coordxy = coordinate(High=720, Width=1280).permute(1, 2, 0).numpy()
coordpatchs = patchify(coordxy, (64,64, 2), step=64)
coordxy = torch.from_numpy(coordpatchs).permute(0,1,2,5,3,4)



def datasetemp(datapath, camframeperepoch):
    listofcam = cfg.rm_ds(os.listdir(datapath))
    patches = [f'patch_{i}_{j}' for i in range(11) for j in range(20)]
    temp = dict()
    for patch in patches:
        camtemp = []
        for cam in listofcam:
            campatchpath = os.path.join(datapath, cam, patch)
            listofpatches = os.listdir(campatchpath)
            sublistofpatches = random.sample(listofpatches, camframeperepoch)
            for subpatch in sublistofpatches:
                subpatchpath = os.path.join(campatchpath, subpatch)
                camtemp.append(subpatchpath)
        
        temp[patch] = camtemp

    return temp



class VideoNoiseDataset(Dataset):
    """
    doc
    """
    def __init__(self, datapath, batch_size, numcams, coordaware=False) -> None:
        super().__init__()
        self.cw = coordaware
        self.path = datapath
        self.bs = batch_size
        self.patchs = datasetemp(datapath=datapath, camframeperepoch=batch_size//numcams)
        # print(self.patchs)
        self.patchkeys = list(self.patchs.keys())
        self.xy = coordinate(High=64, Width=64)

    def __len__(self):
        return len(self.patchkeys)

    def getpatch(self, idx):
        patchid = self.patchkeys[idx]
        if self.cw:
            _, hi, wi = patchid.strip('_')
            hi, wi = int(hi), int(wi)
            patchcoord = coordxy[hi, wi, 0]
            patchspaths = self.patchs[patchid]
            patch0 = torch.from_numpy(cv2.imread(patchspaths[0])).permute(2, 0, 1)[1:2, :, :]
            Patchcoord = torch.cat((patch0, patchcoord), dim=0).unsqueeze(dim=0)
            for i in range(1, len(patchspaths)):
                patchi = torch.from_numpy(cv2.imread(patchspaths[i])).permute(2, 0, 1)[1:2, :, :]
                patchi = torch.cat((patchi, patchcoord), dim=0).unsqueeze(dim=0)
                Patchcoord = torch.cat((Patchcoord, patchi), dim=0)

        else:
            patchspaths = self.patchs[patchid]
            Patchcoord = torch.from_numpy(cv2.imread(patchspaths[0])).permute(2, 0, 1)[1:2, :, :]
            Patchcoord = Patchcoord.unsqueeze(dim=0) 
            for i in range(1, len(patchspaths)):
                patchi = torch.from_numpy(cv2.imread(patchspaths[i])).permute(2, 0, 1)[1:2, :, :]
                Patchcoord = torch.cat((Patchcoord, patchi.unsqueeze(dim=0)), dim=0)

        return Patchcoord


    def __getitem__(self, index):
        X = self.getpatch(index)
        return X.float().to(dev)




def create_loader(batch_size=200, caware=False):
    traindata = VideoNoiseDataset(datapath=cfg.paths['train'], batch_size=batch_size, numcams=40, coordaware=caware)
    valdata = VideoNoiseDataset(datapath=cfg.paths['val'], batch_size=batch_size, numcams=5, coordaware=caware)
    return DataLoader(traindata, batch_size=1), DataLoader(valdata, batch_size=1)


def main():
    dpath = cfg.paths['val']
   
    print(coordxy[0,0,0].shape)
    




if __name__ == '__main__':
    main()

            


            


















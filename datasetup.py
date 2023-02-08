import conf as cfg
import torch
from torch.utils.data import Dataset, DataLoader
import os, random
import cv2
import numpy as np





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

coordxy = coordinate(High=1080, Width=1920)



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
        print(self.patchs)
        self.patchkeys = list(self.patchs.keys())
        self.xy = coordinate(High=64, Width=64)

    def __len__(self):
        return len(self.patchkeys)


    def __getitem__(self, index):
        patchid = self.patchkeys[index]
        patchepaths = self.patchs[patchid]
        Patchs = torch.from_numpy(cv2.imread(patchepaths[0])/255.0).permute(2, 0, 1)[1:2, :, :].unsqueeze(dim=0)

        for i in range(1, len(patchepaths)):
            patch = torch.from_numpy(cv2.imread(patchepaths[i])/255.0).permute(2, 0, 1)[1:2, :, :]
            Patchs = torch.cat((Patchs, patch.unsqueeze(dim=0)), dim=0)

        return Patchs.float().to(dev)




def create_loader(batch_size=200, caware=False):
    traindata = VideoNoiseDataset(datapath=cfg.paths['train'], batch_size=batch_size, numcams=40, coordaware=caware)
    valdata = VideoNoiseDataset(datapath=cfg.paths['val'], batch_size=batch_size, numcams=5, coordaware=caware)
    return DataLoader(traindata, batch_size=1), DataLoader(valdata, batch_size=1)


def main():
    dpath = cfg.paths['val']
    # pp = '/Users/hamzeasadi/python/videonoiseprint/data/asqar'
    # r = datasetemp(datapath=pp, camframeperepoch=2)
    # data = VideoNoiseDataset(datapath=dpath, batch_size=200, numcams=5)
    # print(data[0].shape, data[0].device, data[0].dtype)
    trainl, vall = create_loader(batch_size=200)
    batch = next(iter(trainl))
    print(batch.shape, batch.squeeze(dim=0).shape)
    




if __name__ == '__main__':
    main()

            


            


















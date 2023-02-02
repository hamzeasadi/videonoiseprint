import conf as cfg
import torch
from torch.utils.data import Dataset, DataLoader
import os, random
import cv2
import numpy as np





dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def datatemp(H=64, W=64, href=1080, wref=1920):
    hstart = (href%64)//2
    wstart = 0
    numh = href//64
    numw = wref//64
    tmp = dict()
    patch_cnt = 0
    for i in range(numh):
        hi = hstart + i*H
        for j in range(numw):
            wi = wstart + j*W
            tmp[f'patch{patch_cnt}'] = (hi, wi)
            patch_cnt+=1
    return tmp


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



def cropimg(img, patchid, H=64, W=64, coordinate=False):
    h, w = patchid
    crop = img[h:h+H, w:w+W, 1:2]
    crop = ((crop - np.min(crop))/(np.max(crop) - np.min(crop) + 0.00001))
    crop = torch.from_numpy(crop).permute(2, 0, 1).float()
    if coordinate:
        coordcrop = coordxy[:, h:h+H, w:w+W]
        crop = torch.cat((crop, coordcrop), dim=0)
    return crop.unsqueeze(dim=0)

def getpatch(folderpath, patchid, numframes, coord=False):
    imgnames = os.listdir(folderpath)
    subimages = random.sample(imgnames, numframes)

    imgpath = os.path.join(folderpath, subimages[0])
    img = cv2.imread(imgpath)
    Crops = cropimg(img=img, patchid=patchid, coordinate=coord)

    for i in range(1, numframes):
        imgpath = os.path.join(folderpath, subimages[i])
        img = cv2.imread(imgpath)
        crop = cropimg(img=img, patchid=patchid, coordinate=coord)
        Crops = torch.cat((Crops, crop), dim=0)
    
    return Crops




class VideoNoiseDataset(Dataset):
    """
    doc
    """
    def __init__(self, datapath, batch_size, coordaware=False) -> None:
        super().__init__()
        self.bs = batch_size
        self.cw = coordaware
        self.path = datapath
        self.batchids = datatemp()
        keylist = list(self.batchids.keys())
        self.bidkeys = random.sample(keylist, len(keylist))
        folderspath = [os.path.join(datapath, foldername) for foldername in cfg.rm_ds(os.listdir(datapath))]
        self.folders = random.sample(folderspath, len(folderspath))
        self.imgpercam = batch_size//len(folderspath)

    def __len__(self):
        return len(self.bidkeys)

    def __getitem__(self, index):
        patches = getpatch(folderpath=self.folders[0], patchid=self.batchids[self.bidkeys[index]], numframes=self.imgpercam, coord=self.cw)
        for f in range(1, len(self.folders)):
            patch = getpatch(folderpath=self.folders[f], patchid=self.batchids[self.bidkeys[index]], numframes=self.imgpercam, coord=self.cw)
            patches = torch.cat((patches, patch), dim=0)

        return patches.to(dev)

def create_loader(batch_size=198, caware=False):
    traindata = VideoNoiseDataset(datapath=cfg.paths['train'], batch_size=batch_size, coordaware=caware)
    valdata = VideoNoiseDataset(datapath=cfg.paths['val'], batch_size=batch_size, coordaware=caware)
    return DataLoader(traindata, batch_size=1), DataLoader(valdata, batch_size=1)


def main():
    dpath = cfg.paths['train']
    dataset = VideoNoiseDataset(datapath=dpath, batch_size=20, coordaware=True)
    batch = dataset[0]
    print(batch.shape)



if __name__ == '__main__':
    main()

            


            


















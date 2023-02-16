import conf as cfg
import torch
from torch.utils.data import Dataset, DataLoader
import os, random
import cv2
import numpy as np
from torchvision import transforms




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def datasetemp(datapath):
    listofcam = cfg.rm_ds(os.listdir(datapath))
    patches = [f'patch_{i}_{j}' for i in range(11) for j in range(20)]
    temp = []
    for cam in listofcam:
        for patch in patches:
            patchpath = os.path.join(datapath, cam, patch)
            temp.append(patchpath)
    return temp





class VideoNoiseDataset(Dataset):
    def __init__(self, datapath, coordaware=False) -> None:
        super().__init__()
        self.tmp = datasetemp(datapath)
        self.trf = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Grayscale()])
    def __len__(self):
        return int(1e+5)


    def __getitem__(self, index):
        out = random.choices([1,3,0,1,0,2,0,0], k=1)

        if out==1:
            imgs = random.sample(self.tmp, 1)
            imglist = random.sample(os.listdir(imgs[0]), 2)

            img1rand = imglist[0]
            img2rand = imglist[1]

            img01 = cv2.imread(os.path.join(imgs[0], img1rand))
            img1 = self.trf((img01-127)/255)


            img02 = cv2.imread(os.path.join(imgs[1], img2rand))
            img2 = self.trf((img02-127)/255)

            y = torch.tensor([1], device=dev, dtype=torch.float32)

        else:

            imgs = random.sample(self.tmp, 2)
            img1rand = random.sample(os.listdir(imgs[0]), 1)
            img2rand = random.sample(os.listdir(imgs[1]), 1)

            img01 = cv2.imread(os.path.join(imgs[0], img1rand[0]))
            img1 = self.trf((img01-127)/255)


            img02 = cv2.imread(os.path.join(imgs[1], img2rand[0]))
            img2 = self.trf((img02-127)/255)

            y = torch.tensor([0], device=dev, dtype=torch.float32)
    

        return img1.float(), img2.float(), y




def create_loader(batch_size=200, caware=False):
    traindata = VideoNoiseDataset(datapath=cfg.paths['train'], coordaware=caware)
    valdata = VideoNoiseDataset(datapath=cfg.paths['val'], coordaware=caware)
    return DataLoader(traindata, batch_size=batch_size), DataLoader(valdata, batch_size=100)


def main():
    dpath = os.path.join(cfg.paths['data'], 'asqar')
    
    # data = VideoNoiseDataset(datapath=dpath)
    # print(len(data))
    # X = data[0]
    for i in range(10):
        
        print(out[0])
    




if __name__ == '__main__':
    main()

            


            


















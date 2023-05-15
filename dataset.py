import os
import random
from itertools import combinations
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import torch
from conf import Paths
import utils

paths = Paths()

input_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

def create_train_loader(batch_size:int=128):
    dataset = ImageFolder(paths.server_train_path, transform=input_transform)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=22)
    return loader


class Eval_vision(Dataset):

    def __init__(self, data_path:str=paths.server_test_path) -> None:
        super().__init__()
        self.data_pairs = utils.pairs(path=data_path)
        self.list_pairs = list(self.data_pairs.keys())

    def __len__(self):
        return len(self.list_pairs)
    
    def __getitem__(self, index):
        mypair = self.data_pairs[index]
        img1 = cv2.imread(mypair[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(mypair[1], cv2.IMREAD_GRAYSCALE)

        img1t = torch.from_numpy(img1/255).unsqueeze(dim=0).unsqueeze(dim=0).float()
        img2t = torch.from_numpy(img2/255).unsqueeze(dim=0).unsqueeze(dim=0).float()
        return torch.concat((img1t, img2t), dim=1), torch.tensor(mypair[2])
    

        


    






if __name__ == '__main__':
    print(os.path.basename(__file__))

    dataset = Eval_vision()
    x, y = dataset[0]
    print(x)
    print(y)
    print(x.shape)

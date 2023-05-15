import os
import random
from itertools import combinations
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from conf import Paths


paths = Paths()

input_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

def create_train_loader(batch_size:int=128):
    dataset = ImageFolder(paths.server_train_path, transform=input_transform)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=22)
    return loader




        


    






if __name__ == '__main__':
    print(os.path.basename(__file__))

    x = [1,2,3,4]
    comb = list(combinations(x, 2))
    print(comb)

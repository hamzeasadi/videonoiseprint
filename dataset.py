import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from conf import Paths


paths = Paths()

input_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

def create_train_loader(batch_size:int=128):
    dataset = ImageFolder(paths.server_train_path, transform=input_transform)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=12)
    return loader



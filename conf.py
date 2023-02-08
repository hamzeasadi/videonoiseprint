import os
import random
from os.path import expanduser


home = expanduser("~")
root = os.getcwd()
data = os.path.join(root, 'data')
paths = dict(
    root=root, data=data, model=os.path.join(data, 'model'), 
    dataset=os.path.join(home, 'project', 'Datasets'),
    train=os.path.join(data, 'iframes', 'train'), val=os.path.join(data, 'iframes', 'val'), 
    
    )

def rm_ds(array):
    try:
        array.remove('.DS_Store')
    except Exception as e:
        print(e)
    return array


def createdir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(e)



def main():
    for k, v in paths.items():
        createdir(path=v)




if __name__ == '__main__':
    main()
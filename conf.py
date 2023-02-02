import os
import random


root = os.getcwd()
data = os.path.join(root, 'data')
paths = dict(
    root=root, data=data, model=os.path.join(data, 'model'),
    train=os.path.join(data, 'iframes', 'train'), val=os.path.join(data, 'iframes', 'val') 
    )

def rm_ds(array):
    try:
        array.remove('.DS_Store')
    except Exception as e:
        print(e)
    return array



def main():
    pass




if __name__ == '__main__':
    main()
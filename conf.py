import os
import random


root = os.getcwd()
data = os.path.join(root, 'data')
paths = dict(
    root=root, data=data, model=os.path.join(data, 'model'), 
    train=os.path.join(data, 'iframes', 'train'), val=os.path.join(data, 'iframes', 'val'), 
    testing=os.path.join(data, 'testing'), 
    cams=os.path.join(data, 'testing', 'cams'), refs=os.path.join(data, 'testing', 'refs'), 
    camsvideos=os.path.join(data, 'testing', 'videos'),
    np=os.path.join(data, 'testing', 'np'), model1=os.path.join(data, 'testing', 'model1'), model2=os.path.join(data, 'testing', 'model2'),
    model1out=os.path.join(data, 'testing', 'model1out'), model2out=os.path.join(data, 'testing', 'model2out')
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
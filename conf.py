import os
from typing import NamedTuple


class Paths(NamedTuple):
    
    root:str = os.getcwd()
    data:str = os.path.join(root, 'data')
    dataset:str = os.path.join(data, 'dataset')
    model: str = os.path.join(data, 'model')
    result:str = os.path.join(data, 'result')
    config:str = os.path.join(data, 'config')

    local_train_path:str = os.path.join(dataset, 'train')
    local_test_path:str = os.path.join(dataset, 'test')

    server_train_path = os.path.join(os.path.expanduser('~'), 'project', 'Datasets', 'Vision_450x450', 'train')
    server_test_path = os.path.join(os.path.expanduser('~'), 'project', 'Datasets', 'Vision_450x450', 'test')




    @staticmethod
    def create_dir(path:str):
        if not os.path.exists(path):
            os.makedirs(path)
    @staticmethod
    def init_paths():
        for path in Paths():
            Paths.create_dir(path)



if __name__ == '__main__':
    print(__file__)
    Paths.init_paths()

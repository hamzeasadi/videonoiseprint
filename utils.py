import os
import random
from itertools import combinations

from conf import Paths

paths = Paths()

def pairs(path:str, max_pair:int=15):
    cams = [f for f in os.listdir(path) if f.startswith('D')]
    num_cams = len(cams)
    total_pairs = dict()
    pair_cnt = 0

    for cam in cams:
        sub_sample = random.sample([f for f in os.listdir(os.path.join(path, cam)) if f.endswith('.png')], max_pair)
        comb = list(combinations(sub_sample, 2))
        for com in comb:
            path1 = os.path.join(path, cam, com[0])
            path2 = os.path.join(path, cam, com[1])
            total_pairs[pair_cnt] = (path1, path2, 1)
            pair_cnt += 1

    num_neg = len(comb)*num_cams
    ops_cams = list(combinations(cams, 2))
    frprcm = num_neg//len(ops_cams)
    for npair in ops_cams:
        cam_1_data = random.sample([f for f in os.listdir(os.path.join(path, npair[0])) if f.endswith('.png')], frprcm)
        cam_2_data = random.sample([f for f in os.listdir(os.path.join(path, npair[1])) if f.endswith('.png')], frprcm)
        for i in range(len(cam_1_data)):
            path1 = os.path.join(path, npair[0], cam_1_data[i])
            path2 = os.path.join(path, npair[1], cam_2_data[i])
            total_pairs[pair_cnt] = (path1, path2, 1)
            pair_cnt += 1


    return total_pairs



if __name__ == '__main__':
    print(__file__)

    pair_dict = pairs(path=paths.server_test_path, max_pair=15)
    print(pair_dict)
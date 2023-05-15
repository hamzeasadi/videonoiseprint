import os
import random
from itertools import combinations

def pairs(path:str, max_pair:int=15):
    cams = [f for f in os.listdir(path) if f.startswith('D')]
    num_cams = len(cams)
    
    pos_pairs = dict()
    neg_pairs = dict()
    for cam in cams:
        sub_sample = random.sample([f for f in os.listdir(os.path.join(path, cam)) if f.endswith('.png')], max_pair)
        comb = list(combinations(sub_sample, 2))
        pos_pairs[cam] = comb
    
    num_neg = len(comb)*num_cams
    ops_cams = list(combinations(cams, 2))
    frprcm = num_neg//len(ops_cams)
    for npair in ops_cams:
        cam_1_data = random.sample([f for f in os.listdir(os.path.join(path, npair[0])) if f.endswith('.png')], frprcm)
        cam_2_data = random.sample([f for f in os.listdir(os.path.join(path, npair[1])) if f.endswith('.png')], frprcm)
        neg_pairs[npair] = (cam_1_data, cam_2_data)

    return pos_pairs, neg_pairs
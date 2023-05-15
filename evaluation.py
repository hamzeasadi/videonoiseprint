import os
import argparse
import torch
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import model as m
from conf import Paths
import dataset as dst



parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description='eval config')
parser.add_argument('--ckpoint_num', '-ckpn', type=int, required=True)
# parser.add_argument('--res', action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()


paths = Paths()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dev = torch.device('cpu')


def cosime_score(tensor_pair):
    X = tensor_pair.cpu().detach().numpy()
    score = (np.dot(X[0],X[1]))/(np.linalg.norm(X[0])*np.linalg.norm(X[1]))
    return min(1, max(0, score))


if __name__ == '__main__':
    print(__file__)
    dataset = dst.Eval_vision()
    num_pairs = len(dataset)

    save_path = os.path.join(paths.result, f'ckpoint_{args.ckpoint_num}')
    paths.create_dir(save_path)

    model = m.ConstLayer(ks=5, inch=1, outch=3, num_classes=28, dev=dev)
    model.to(dev)
    ckpoint = torch.load(os.path.join(paths.model, f'ckpoint_{args.ckpoint_num}.pt'), map_location=dev)
    model.load_state_dict(ckpoint)

    model.resnet.fc = nn.Identity()
    model.to(dev)
    model.eval()
    y_t = []
    y_p = []
    # with torch.no_grad():
    #     for i in range(num_pairs):
    #         x1x2, lbl = dataset[i]
    #         out1, out2, out3 = model(x1x2.to(dev))
    #         y_t.append(lbl)
    #         score = cosime_score(out1)
    #         y_p.append(score)

    # precision, recall, thresholds = precision_recall_curve(y_t, y_p)
    # auc = roc_auc_score(y_t, y_p)

    # print(f'auc={auc}')

    constlayer = model.constlayer
    x1x2, lbl = dataset[10]
    out = constlayer(x1x2.to(dev))
    out1 = out[0]
    out = out1.cpu().detach().numpy()
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(x1x2[0].squeeze().numpy(), cmap='gray')
    axs[1].imshow(out[0], cmap='gray')
    axs[2].imshow(out[1], cmap='gray')
    axs[3].imshow(out[2], cmap='gray')
    plt.savefig(os.path.join(save_path, 'res.png'))













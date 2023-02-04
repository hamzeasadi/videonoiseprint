import os
import conf as cfg
import torch
import cv2
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import utils
import helper as hp

















def main():
    x = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    y = np.random.randint(low=0, high=2, size=x.shape[0])
    tpr, fpr, thresholds = roc_curve(x, y)
    print(tpr)
    print(fpr)
    print(thresholds)



if __name__ == "__main__":
    main()


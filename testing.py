import os
import conf as cfg
import torch
import cv2
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import utils
import helper as hp






def ncc_cams(srcnps, refnps):
    # srcnps refere to nps for all cameras, trgnps referes to reference nps for each camera
    trgcams = os.listdir(srcnps)
    trgcams = cfg.rm_ds(trgcams)

    listofrefcamsnps = os.listdir(refnps)
    listofrefcamsnps = cfg.rm_ds(listofrefcamsnps)
    all_ncc = dict()
    all_mse = ()
    for refnpcam in listofrefcamsnps:
        refsigpath = os.path.join(refnps, refnpcam)
        refsig = np.load(refsigpath)

        for trgcam in trgcams:
            trgcampath = os.path.join(srcnps, trgcam)
            trgsignals = os.listdir(trgcampath)
            nccs = []
            mses = []
            for trgsignal in trgsignals:
                trgsignalpath = os.path.join(trgcampath, trgsignal)
                trgsig = np.load(trgsignalpath)
                nccs.append(hp.NCC(refnp=refsig, testnp=trgsig))
                mses.append(hp.meanse(refnp=refsig, testnp=trgsig))
            all_ncc[(refnpcam, trgcam)] = nccs
            all_mse[(refnpcam, trgcam)] = mses

    return all_ncc, all_ncc













def main():
    x = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    
    srcnps = cfg.paths['np']
    refnps = cfg.paths['refs']
    allncc, allmse = ncc_cams(srcnps=srcnps, refnps=refnps)


if __name__ == "__main__":
    main()


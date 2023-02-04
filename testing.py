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
    print(trgcams)
    listofrefcamsnps = os.listdir(refnps)
    listofrefcamsnps = cfg.rm_ds(listofrefcamsnps)
    all_ncc = dict()
    all_mse = dict()
    for refnpcam in listofrefcamsnps:
        refsigname = os.listdir(os.path.join(refnps, refnpcam))
        refsigpath = os.path.join(refnps, refnpcam, refsigname)
        
        refsig = np.load(refsigpath)

        for trgcam in trgcams:
            trgcampath = os.path.join(srcnps, trgcam)
            trgsignals = os.listdir(trgcampath)
            nccs = []
            mses = []
            for trgsignal in trgsignals:
                trgsignalpath = os.path.join(trgcampath, trgsignal)
                trgsig = np.load(trgsignalpath)
                nccs.append(hp.NCC(refnp=torch.from_numpy(refsig), testnp=torch.from_numpy(trgsig)))
                mses.append(hp.meanse(refnp=torch.from_numpy(refsig), testnp=torch.from_numpy(trgsig)))
            all_ncc[(refnpcam, trgcam)] = nccs
            all_mse[(refnpcam, trgcam)] = mses

    return all_ncc, all_ncc













def main():
    x = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    
    srcnps = cfg.paths['np']
    refnps = cfg.paths['refs']
    allncc, allmse = ncc_cams(srcnps=srcnps, refnps=refnps)
    print(allncc)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(allmse)

if __name__ == "__main__":
    main()


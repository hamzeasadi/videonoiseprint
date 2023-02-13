import os
import conf as cfg
import torch
import cv2
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import utils
import helper as hp
import model as m
from matplotlib import pyplot as plt
from torch import nn as nn




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
        refnpcappath = os.path.join(refnps, refnpcam)
        refsigname = os.listdir(refnpcappath)[0]
        refsigpath = os.path.join(refnpcappath, refsigname)
        
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

    return all_ncc, all_mse













def main():
    img = cv2.imread(os.path.join(cfg.paths['data'], 'video1iframe0.bmp'))
    # img0 = 2*(img[300:700, 850:1250, 1:2] - np.min(img[:, :, 1:2] ))/(np.max(img[:, :, 1:2] ) - np.min(img[:, :, 1:2] ) + 1e-5) -1
    img0 = 1*(img[:, :, 1:2] - np.min(img[:, :, 1:2] ))/(np.max(img[:, :, 1:2] ) - np.min(img[:, :, 1:2] ))
    
    # img0 = (img[300:700, 850:1250, 1:2] - 127 )/255
    imgt = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(dim=0).float()


    kt = utils.KeepTrack(path=cfg.paths['model'])
    listofmodels = os.listdir(cfg.paths['model'])
    # state = kt.load_ckp(fname=listofmodels[-1])
    # state = kt.load_ckp(fname=f'noisprintcoord2_{50}.pt')
    state = kt.load_ckp(fname='noisprintcoord2_99.pt')
    print(state['trainloss'], state['valloss'])
    model = m.VideoPrint(inch=1, depth=15)
    model = nn.DataParallel(model)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(imgt)
        print(out.shape)
    
    img1 = out.detach().squeeze().numpy()
    plt.imshow(img1, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()


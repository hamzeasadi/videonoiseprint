import os, random
import conf as cfg
import torch
import cv2
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import model as m


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def iframe_extract(srcdatapath, trgdatapath):
    listofcams = os.listdir(srcdatapath)
    listofcams = cfg.rm_ds(listofcams)
    for i, cam in enumerate(listofcams):
        campath = os.path.join(srcdatapath, cam)
        listofcamvidoes = os.listdir(campath)
        listofcamvidoes = cfg.rm_ds(listofcamvidoes)
        # cam10subvideos = random.sample(listofcamvidoes, 10)
        for j, camvideo in enumerate(listofcamvidoes):
            trgvideopath = os.path.join(trgdatapath, cam, f'video_{j}')
            cfg.createdir(trgvideopath)
            videopath = os.path.join(campath, camvideo)
            command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync vfr -frame_pts true -x264opts no-deblock {trgvideopath}/iframe%d.bmp" 
            os.system(command=command)


def coordinate(High, Width):
    xcoord = torch.ones(size=(High, Width), dtype=torch.float32)
    ycoord = torch.ones(size=(High, Width), dtype=torch.float32)
    for i in range(High):
        xcoord[i, :] = 2*(i/High) - 1
    for j in range(Width):
        ycoord[:, j] = 2*(j/Width) - 1
    
    coord = torch.cat((xcoord.unsqueeze(dim=0), ycoord.unsqueeze(dim=0)), dim=0)
    return coord




def central_crop(imgpath, H=720, W=1280, coord=False):
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    hc = h//2
    wc = w//2
    cnt_crop = img[hc-H//2:hc-H//2, wc-W//2:w+W//2, 1:2]
    cnt_crop = 2*((cnt_crop - np.min(cnt_crop))/(np.max(cnt_crop) - np.min(cnt_crop))) - 1
    cnt_crop = torch.from_numpy(cnt_crop).permute(2, 0, 1).float()
    if coord:
        coordxy = coordinate(High=1080, Width=1920)
        coords = coordxy[:, 1080//2 - H//2:1080//2 + H//2, 1920//2 - W//2:1920//2 + W//2]
        cnt_crop = torch.cat((cnt_crop, coords), dim=0)
    
    return cnt_crop.unsqueeze(dim=0)


def videonp_calc(net:nn.Module, videoiframepath, numframe, cw, method='avg'):
    iframes = os.listdir(videoiframepath)
    iframes = cfg.rm_ds(iframes)
    n = min(numframe, len(iframes))
    subiframes = random.sample(iframes, n)
    noiseprint = torch.zeros(size=(1, 720, 1280), device=dev, dtype=torch.float32)
    net.to(dev)
    net.eval()
    with torch.no_grad():
        for iframe in subiframes:
            iframepath = os.path.join(videoiframepath, iframe)
            crop = central_crop(imgpath=iframepath, coord=cw)
            vidnp = net(crop)
            noiseprint += vidnp/numframe
    
    return noiseprint


def reference_calculate(net:nn.Module, campath:str, iframpervideo, numvideos, method='avg', coordinate=False):
    # [vidoe1iframs, video2ifram, ..]
    camvideosiframe = os.listdir(campath)
    camvideosiframe = cfg.rm_ds(camvideosiframe)
    subcamvideosiframe = random.sample(camvideosiframe, numvideos)

    fingerprint = torch.zeros(size=(1, 720, 1280), device=dev, dtype=torch.float32)

    for camvideoiframe in subcamvideosiframe:
        # [vidoe1ifram/iframe1, video1ifram2, ..]
        camvideoiframepath = os.path.join(campath, camvideoiframe)
        camnp = videonp_calc(net=net, videoiframepath=camvideoiframepath, numframe=iframpervideo, cw=coordinate, method=method)
        fingerprint += camnp/numvideos

    return fingerprint


def NCC(refnp, testnp):
    ref_mean = refnp.mean()
    ref_std = refnp.std()
    ref_norm = (refnp - ref_mean)/ref_std

    test_mean = testnp.mean()
    test_std = testnp.std()
    test_norm = (testnp - test_mean)/test_std

    ref_test_ncc = F.conv2d(ref_norm.unsqueeze(dim=0), test_norm.unsqueeze(dim=0).flip(dims=(2, 3)), padding=0)
    return ref_test_ncc


def meanse(refnp, testnp):
    diff = torch.subtract(refnp, testnp)
    return torch.linalg.matrix_norm(diff)

def save_np(noiseprint, numiframe, camid, videoid, trgpath):
    noiseprint_np = noiseprint.numpy()
    path = os.path.join(trgpath, camid)
    cfg.createdir(path=path)
    filename = f'np_{videoid}_{numiframe}.npy'
    np.save(os.path.join(path, filename), noiseprint_np)


def save_all_ref(Net:nn.Module, data_path:str, iframepervideo, numvideos, method='avg', cw=False):
    cams = os.listdir(data_path)
    cams = cfg.rm_ds(cams)
    for cam in cams:
        campath = os.path.join(data_path, cam)
        refnoise = reference_calculate(net=Net, campath=campath, iframpervideo=iframepervideo, numvideos=numvideos, method=method, coordinate=cw)
        save_np(noiseprint=refnoise, numiframe=numvideos*iframepervideo, camid=cam, videoid='ref', trgpath=cfg.paths['refs'])

def cam_noiseprint(net:nn.Module, campath, framepervideo, coordaware, method):
    camname = campath.split('/')[-1].strip()
    camvideosiframe = os.listdir(campath)
    camvideosiframe = cfg.rm_ds(camvideosiframe)
    for videoiframe in camvideosiframe:
        videoiframepath = os.path.join(campath, videoiframe)
        videonp = videonp_calc(net=net, videoiframepath=videoiframepath, numframe=framepervideo, cw=coordaware, method=method)
        savepath = os.path.join(cfg.paths['np'], camname, f'{videoiframe}_{framepervideo}.npy')
        np.save(savepath, videonp.numpy())

def all_cams_noisprint(net:nn.Module, camspaths, framepervideo, coordaware, method):
    cams = os.listdir(camspaths)
    cams = cfg.rm_ds(cams)
    for cam in cams:
        campath = os.path.join(camspaths, cam)
        cam_noiseprint(net=net, campath=campath, framepervideo=framepervideo, coordaware=coordaware, method=method)
    


    






def main():
    # srcpath = cfg.paths['camsvideos']
    # trgpath = cfg.paths['camsiframes']
    # # iframe_extract(srcdatapath=srcpath, trgdatapath=trgpath)


    inch=1
    coord = False
    if coord:
        inch=3
    model = m.VideoPrint(inch=inch, depth=15)
    state = torch.load(cfg.paths['model1'], map_location=dev)
    model.load_state_dict(state['model'])
    camerapaths = cfg.paths['camsiframes']
    framepervideo = 10
    all_cams_noisprint(net=model, camspaths=camerapaths, framepervideo=framepervideo, coordaware=coord, method='avg')


if __name__ == '__main__':
    main()
import conf as cfg 
import os 
import cv2 
import numpy as np 
# from PIL import Image
# from patchify import patchify
 
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
             
            
def video_iframes(srcvideopath, trgiframepath, cntr): 
    videoname = srcvideopath.split('/')[-1].strip() 
    try: 
        command = f"ffmpeg -skip_frame nokey -i {srcvideopath} -vsync vfr -frame_pts true -x264opts no-deblock {trgiframepath}/iframe{cntr}_%d.bmp" 
        os.system(command=command) 
    except Exception as e: 
        print(e) 
 
def camiframes(campath, camtrgiframpath): 
    listofvideos = os.listdir(campath) 
    listofvideos = cfg.rm_ds(listofvideos)  
    for k, videoname in enumerate(listofvideos): 
        videopath = os.path.join(campath, videoname) 
        video_iframes(srcvideopath=videopath, trgiframepath=camtrgiframpath, cntr=k) 
 
def central_crop_rotate(img, H=720, W=1280): 
    h, w, c = img.shape 
    hc, wc = h//2, w//2 
    hi, wi = H//2, W//2 
    if h>w: 
        imgt = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        return imgt[wc-hi:wc+hi, hc-wi:hc+wi, :] 
    else: 
        return img[hc-hi:hc+hi, wc-wi:wc+wi, :] 
 
def cam_align(camdirpath): 
    camiframelist = os.listdir(camdirpath) 
    camiframelist = cfg.rm_ds(camiframelist) 
    for camiframe in camiframelist: 
        camiframepath = os.path.join(camdirpath, camiframe) 
        img1 = cv2.imread(camiframepath) 
        # if img1 is not None: 
        try: 
            img1.shape 
            img2 = central_crop_rotate(img=img1) 
            cv2.imwrite(camiframepath, img2) 
        except Exception as e: 
            print(camdirpath) 
 
def dataset_iframes(srccamspath, trgcamsiframspath): 
    listofcams = os.listdir(srccamspath) 
    listofcams = cfg.rm_ds(listofcams) 
    for camname in listofcams: 
        camerapath = os.path.join(srccamspath, camname) 
        camiframepath = os.path.join(trgcamsiframspath, camname) 
        # cfg.createdir(camiframepath) 
         
        # camiframes(campath=camerapath, camtrgiframpath=camiframepath) 
        cam_align(camdirpath=camiframepath) 


def patchify(img, patchsize, step):
    h, w, c = img.shape
    H, W, C = patchsize
    numh = h//H
    numw = w//W
    patcgholder = np.zeros(shape=(numh, numw, 1, H, W, C))
    for i in range(numh):
        hi = i*H
        for j in range(numw):
            wi = j*W
            patch = img[hi:hi+H, wi:wi+W, :]
            patcgholder[i, j, 0] = patch

    return patcgholder



def patching(srcpath, trgpath): 
    listofcams = cfg.rm_ds(os.listdir(srcpath)) 
    for cam in listofcams: 
        campath = os.path.join(srcpath, cam) 
        listofframes = cfg.rm_ds(os.listdir(campath)) 
        trgcam = os.path.join(trgpath, cam) 
        cfg.createdir(trgcam) 
        for fr, iframe in enumerate(listofframes): 
            iframepath = os.path.join(campath, iframe) 
            # image = Image.open(iframepath) 
            # image = np.asarray(image) 
            image = cv2.imread(iframepath)
            patches = patchify(image, (64, 64, 3), step=64) 
            patcheshape = patches.shape 
            for i in range(patcheshape[0]): 
                for j in range(patcheshape[1]): 
                    patch = patches[i, j, 0] 
                    patchid = f'patch_{i}_{j}' 
                    # patch = Image.fromarray(patch) 
                    patchpath = os.path.join(trgcam, patchid) 
                    cfg.createdir(patchpath) 
                    # patch.save(os.path.join(patchpath, f'patch{fr}.bmp'))
                    cv2.imwrite(os.path.join(patchpath, f'patch{fr}.bmp'), patch)

             
             
def main(): 
    print(42) 
    srcpath = os.path.join(cfg.paths['data'], 'iframe40cam')
    trgpath = os.path.join(cfg.paths['data'], 'iframes')
    patching(srcpath=srcpath, trgpath=trgpath)
    # patcgholder = np.zeros(shape=(2, 3, 1, 5, 5, 2))
    # print(patcgholder.shape)
     
     
if __name__ == '__main__': 
    main()
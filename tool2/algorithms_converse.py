import time
import sys
import os
import numpy as np
import torch
import cv2
import random
from PIL import Image
from scipy import signal
from matplotlib import pyplot as plt
from tool.utils import makedir, get_gaussian_kernel


# sys.path.append(r"D:\AI-046\Projects\woods03\woods03codes\augment\treats")
# from algorithms
def gaussian_blur(img, size=3):
    gauss_kernel = get_gaussian_kernel(size)
    # print(gauss_kernel)i
    # print(np.sum(gauss_kernel))
    img2 = signal.convolve2d(img, gauss_kernel, "same")
    img2 = np.clip(img2, 0, 255)
    img2 = img2.astype(np.uint8)
    return img2

def img_converse1(img, anglerange=10):
    h, w = img.shape
    funcdict = {0: rotate, 1: flip, 2: remap, 3: perspective}
    funcprob = torch.Tensor([0.2,0.4,0.2,0.2])

    funcnum = np.random.randint(1, len(funcdict)+1)
    # print(funcnum)
    img2 = img.copy()
    # funcids = torch.randperm(len(funcdict))[:funcnum]
    # print(funcids)
    for i in range(funcnum):
        # print(funcid)
        # funcid = funcid.item()
        funcid = torch.multinomial(funcprob, 1).item()

        if funcid==0:

            angle = np.random.randint(-anglerange, anglerange)
            scale = np.random.uniform(0.8, 1.2)
            img2 = rotate(img2, angle, scale)
        elif funcid==1:
            flags = np.random.randint(-1, 2)
            img2 = flip(img2, flags)
        elif funcid==2:
            img2 = remap(img2)
        elif funcid==3:
            img2 = perspective(img2)
        else:
            raise IndexError
    # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    img2 = cv2.resize(img2, (w // 2, h // 2))
    img2 = cv2.resize(img2, (w, h))
    return img2

def img_converse2(img, mask):
    assert img.shape==mask.shape
    h, w = img.shape
    # funcdict = {0: rotate, 1: flip, 2: remap, 3: perspective}
    funcdict = {0: rotate, 1: flip}
    funcnum = np.random.randint(1, len(funcdict)+1)
    # print(funcnum)
    img2 = img.copy()
    mask2 = mask.copy()
    funcids = torch.randperm(len(funcdict))[:funcnum]
    # print(funcids)
    for funcid in funcids:
        # print(funcid)
        funcid = funcid.item()
        if funcid==0:
            angle = np.random.randint(-360, 361)
            scale = np.random.uniform(0.8, 1.2)
            img2 = rotate(img2, angle, scale)
            mask2 = rotate(mask2, angle, scale)
        elif funcid==1:
            flags = np.random.randint(-1, 2)
            img2 = flip(img2, flags)
            mask2 = flip(mask2, flags)
        # elif funcid==2:
        #     img2 = remap(img2)
        #     mask2 = remap(mask2)
        # elif funcid==3:
        #     img2 = perspective(img2)
        else:
            raise IndexError
    # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # img2 = cv2.resize(img2, (w // 2, h // 2))
    img2 = cv2.resize(img2, (w, h))
    mask2 = cv2.resize(mask2, (w, h))
    img2 = img2.clip(0, 255).astype(np.uint8)
    mask2 = mask2.clip(0, 255).astype(np.uint8)
    return img2, mask2



def img_converse(img):
    h, w = img.shape
    # funcdict = {0: rotate, 1: flip, 2: remap, 3: perspective}
    funcdict = {0: rotate, 1: flip, 2: remap}
    funcnum = np.random.randint(1, len(funcdict)+1)
    # print(funcnum)
    img2 = img.copy()
    funcids = torch.randperm(len(funcdict))[:funcnum]
    # print(funcids)
    for funcid in funcids:
        # print(funcid)
        funcid = funcid.item()
        if funcid==0:
            angle = np.random.randint(-360, 361)
            scale = np.random.uniform(0.8, 1.2)
            img2 = rotate(img2, angle, scale)
        elif funcid==1:
            flags = np.random.randint(-1, 2)
            img2 = flip(img2, flags)
        elif funcid==2:
            img2 = remap(img2)
        # elif funcid==3:
        #     img2 = perspective(img2)
        else:
            raise IndexError
    # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # img2 = cv2.resize(img2, (w // 2, h // 2))
    img2 = cv2.resize(img2, (w, h))
    return img2

def rotate(img, angle, scale):
    '''
    旋转
    :param imgpath:
    :param angle:
    :param scale:
    :return:
    '''
    # img = cv2.imread(imgpath, 0)
    h = img.shape[0]
    w = img.shape[1]
    A = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=scale)
    img2 = cv2.warpAffine(img, A, (w, h), borderValue=0)
    # img2 = rm_blank(img, img2)
    # img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    # print(img2)
    return img2


def flip(img, flags):
    '''
    翻转
    :param imgpath:
    :param flags (1,0,-1)
    :return:
    '''
    assert flags == 0 or flags == 1 or flags == -1
    # img = cv2.imread(imgpath, 0)
    img2 = cv2.flip(img, flags)
    return img2


def remap(img):
    '''

    :param img:
    :return:
    '''
    # img = cv2.imread(imgpath, 0)
    h = img.shape[0]
    w = img.shape[1]

    mapx = np.arange(w).reshape(1, -1).repeat(h, 0).astype(np.float32)
    mapy = np.arange(h).reshape(-1, 1).repeat(w, 1).astype(np.float32)
    mapx_ = mapx.copy()
    mapy_ = mapy.copy()
    mapx = mapx_ + random.randint(3, 7) * np.sin(mapy_ / random.randint(5, 15))
    mapy = mapy_ + random.randint(3, 7) * np.sin(mapx_ / random.randint(5, 15))
    img2 = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    # img2 = rm_blank(img, img2, 5, False)
    size = 5
    img2 = img2[size:-size,size:-size]
    try:
        img2 = cv2.resize(img2, (w, h))
    except:
        print(img2.shape, w, h)
        exit(-1)
    # img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    return img2


def perspective(img, ra = 8):
    '''

    :param img:
    :return:
    '''
    # img = cv2.imread(imgpath, 0)
    h = img.shape[0]
    w = img.shape[1]

    x1, x3 = np.random.randint(0, w // ra, (2,))
    x2, x4 = np.random.randint(w * (ra-1) // ra, w, (2,))
    y1, y2 = np.random.randint(0, h // ra, (2,))
    y3, y4 = np.random.randint(h * (ra-1) // ra, h, (2,))
    x1_ = min(x1, x2, x3, x4)
    y1_ = min(y1, y2, y3, y4)
    x2_ = max(x1, x2, x3, x4)
    y2_ = max(y1, y2, y3, y4)

    src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float32)
    dst = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)
    p = cv2.getPerspectiveTransform(src, dst)
    r = cv2.warpPerspective(img, p, (w, h), borderValue=(0,0,0))[y1_:y2_, x1_:x2_]
    # r2 = cv2.warpPerspective(img, p, (w, h), borderValue=(100, 100, 0))
    img2 = cv2.resize(r, (w, h))
    # img2 = rm_blank(img, img2)
    # img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    return img2


def rm_blank(srcimg, dstimg, rmval=250, flags=True):
    '''
    大于阈值的位置等于原来的值
    :param srcimg:
    :param dstimg:
    :param thresh:
    :return:
    '''
    if flags:
        mask = dstimg >= rmval
    else:
        mask = dstimg <= rmval
    dstimg[mask] = srcimg[mask]
    return dstimg


if __name__ == '__main__':
    img_path = r".\cat01.jpg"
    # img1_path = r"/augment/saves/Aug1_ROI_yolo01_ng/temp.jpg"
    img = cv2.imread(img_path, 0)
    # h, w = img.shape
    # print(h,w)
    # img1 = cv2.imread(img1_path, 0)
    # img1 = cv2.resize(img1, (w, h))
    # img1 = morph(img1, img, morph_open)
    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)
    # exit()
    # clip 0-254之间
    img = np.clip(img, 0, 254)
    # print(np.max(img))
    # exit()

    # 旋转
    dst = rotate(img, -30, 1)
    # 加高斯模糊
    # # 翻转
    # dst = flip(img, 1)
    # # 扭曲
    # dst = remap(img)
    # # 透射
    # dst = perspective(img)

    # cv2.imshow("dst", dst)
    t1 = time.time()
    for i in range(100):
        img2 = img_converse1(img, anglerange=10)
    t2 = time.time()
    print(t2-t1)


    # h, w = img2.shape

    # img = np.ones_like(img2).astype(np.uint8)*255
    # img2 = rm_blank(img,img2, 10, False)
    # print(np.min(img2))
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    # print(img2)
    print(img)
    cv2.waitKey(0)

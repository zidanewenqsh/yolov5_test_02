#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        get_yolov5_anchor_02.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/3/2 8:56
# @Description:      Main Function:    xxx
# Function List:     hello() -- print helloworld
# History:
#       <author>    <version>   <time>      <desc>
#       wen         ver0_1      2020/12/15  xxx
# ------------------------------------------------------------------
import os
import numpy as np
import torch
import cv2
import time
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from utils.general import xywh2xyxy
import yaml



def mybox_iou(box1, box2):
    box1 = box1[..., None,:]
    # box2 = box2
    # print(box1.shape, box2.shape)
    # print(box1.shape, box2.shape)
    inter = np.minimum(box1,box2).prod(-1)
    union = np.maximum(box1, box2).prod(-1)
    # exit()
    return inter/union

def getlabelinfo(labelpath):
    imgpath = Path(str(labelpath).replace(os.sep+"labels"+os.sep, os.sep+"images"+os.sep, 1)).with_suffix(".jpg")
    with Image.open(imgpath) as img:
        width, height = img.size
        # print(img.size) # (640, 480)
        # img1 = np.array(img) # (480, 640, 3)
        # print(img1.shape)

    with open(labelpath) as f:
        labelarr = np.array([x.split(" ")[3:] for x in f.read().split("\n") if x != '']).astype(np.float32)
        # labelarr = np.array(labelinfo).astype(np.float32)
    # print(f"labelarr1:{labelarr.shape} {imgpath}")
    if len(labelarr)>0:
        labelarr *= np.array([width, height])
    # else:
    #     print(labelarr)
    #     exit()
    # print(f"labelarr2:{labelarr.shape}")
    # exit()
    return labelarr.astype(np.int64) # 返回带宽高的array

def labelarr_nms(labelarr, num):
    if len(labelarr) < num*2:
        return np.array([]), np.array([])

    y_pred = KMeans(n_clusters=num).fit_predict(labelarr)
    # print(y_pred)


    ylist = []

    for i in range(num):
        mask = y_pred == i
        y = labelarr[mask]
        # print(y.shape)
        # print(np.median(y, axis=0).astype(np.int64))
        ylist.append(np.mean(y, axis=0))
    yarr = np.stack(ylist).astype(np.int64)
    # print(yarr)
    biou = mybox_iou(yarr, labelarr)
    # print(biou)
    mask = (biou < iouthresh)
    ids = mask.sum(1).argmin()
    # print(mask)
    # print(ids)
    mask_ = mask[ids]
    # print(mask_)
    # labelarr_ = labelarr[mask_]
    # print(labelarr_.shape)
    return labelarr[mask_], yarr[ids]

def get_boxiou_accuracy(box1, box2, thresh):
    iou = mybox_iou(box1, box2)
    return np.mean(np.max(iou, axis=-1) > thresh)

def get_anchor(labelarr):
    yarrlist = []
    for i in range(12):
        labelarr, yarr = labelarr_nms(labelarr, num=12)
        if len(labelarr) == 0:
            break
        else:
            yarrlist.append(yarr)
    anchor = np.stack(yarrlist)
    return anchor

def print_anchor(anchor_:np.ndarray):
    ids = anchor_.mean(1).argsort()

    anchor = anchor_[ids]
    print(anchor)
if __name__ == '__main__':
    iouthresh = 0.2
    labelarr = np.load("mywoodslabel01.npy")
    labelarr_ = labelarr.copy()
    print(labelarr.shape)
    # 聚类
    # num = 3
    # y_pred = KMeans(n_clusters=num).fit_predict(labelarr)
    # # print(y_pred)
    # ylist = []
    # for i in range(num):
    #     mask = y_pred == i
    #     y = labelarr[mask]
    #     # print(y.shape)
    #     # print(np.median(y, axis=0).astype(np.int64))
    #     ylist.append(np.mean(y, axis=0))
    # yarr = np.stack(ylist).astype(np.int64)
    # print(yarr)
    # biou = mybox_iou(yarr, labelarr)
    # print(biou)
    # mask = (biou<iouthresh)
    # ids = mask.sum(1).argmin()
    # # print(mask)
    # # print(ids)
    # mask_ = mask[ids]
    # # print(mask_)
    # labelarr_ = labelarr[mask_]
    # print(labelarr_.shape)
    anchor = get_anchor(labelarr)
    print_anchor(anchor)
    print(len(anchor))
    accu = get_boxiou_accuracy(anchor, labelarr_, iouthresh)
    print(accu)
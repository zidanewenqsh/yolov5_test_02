#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        get_yolov5_anchor_01.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/3/1 8:56
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
        # exit()
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
# images_dir = Path("../../coco128/images/train2017")
# labels_dir = Path("../../coco128/labels/train2017")

images_dir = Path(r"F:\FProjectsData\woods05\datas\Auguments\saves\Aug_woods_ng_id41\train\images")
labels_dir = Path(r"F:\FProjectsData\woods05\datas\Auguments\saves\Aug_woods_ng_id41\train\labels")
maxnum = 5000
# print(list(labels_dir.glob("*.txt")))
imgfilelist = list(images_dir.glob("*.jpg"))
labelfilelist = list(labels_dir.glob("*.txt"))
maxnum = min(maxnum, len(labelfilelist))
ids = torch.randperm(len(imgfilelist))[:maxnum].tolist()

print(len(imgfilelist))
imgfilelist = np.array(imgfilelist)[ids]
labelfilelist = np.array(labelfilelist)[ids]
# print(len(imgfilelist))
# exit()
# labellist = []
# labeldict = {}
# for labelfile in labelfilelist:
#     with open(labelfile) as f:
#         labellist.extend([x for x in f.read().split("\n") if x !=''])
        # labeldict
# print(labellist)
# print(len(labellist))
labeldict = {x.stem:getlabelinfo(x) for x in labelfilelist if len(getlabelinfo(x))>0}
print(len(labeldict)) # 128->126
# for k,v in labeldict.items():
#     print(v.shape)

labelarr = np.concatenate(list(labeldict.values()))
print(labelarr.shape) # (929, 4)
# print(labelarr)
# np.save("mycocolabel01", labelarr)
np.save("mywoodslabel01", labelarr)
exit(0)
# print(labelarr)
# labelarr = xywh2xyxy(labelarr)
# print(labelarr)
# [[306 330 611 285]
#  [471 118 319 228]
#  [407 351 316 245]
#  ...
#  [219  80  27  66]
#  [332 232 304 244]
#  [321 350 637 141]]
# [[  0 187 611 472]
#  [311   4 630 232]
#  [249 228 565 473]
#  ...
#  [205  47 232 113]
#  [180 110 484 354]
#  [  2 279 639 420]]
label1= labelarr[labelarr[:,1]/labelarr[:,0]>3]
label2= labelarr[labelarr[:,0]/labelarr[:,1]>3]
print(label1.shape, label2.shape)
label3= labelarr[(labelarr[:,0]/labelarr[:,1]<1.2)*(labelarr[:,1]/labelarr[:,0]<1.2)]
# print(label3.shape)
# labelarr = np.concatenate((label1, label2, label3), axis=0)
# print(labelarr.shape)
# l1 = labelarr[:,0]/labelarr[:,1]<2
# l2 = labelarr[:,1]/labelarr[:,0]<2
# print(l1)
# print(l2)
# print(l1 * l2)
num = 9
y_pred = KMeans(n_clusters=num).fit_predict(labelarr)
ylist = []
for i in range(num):
    mask = y_pred == i
    y = labelarr[mask]
    # print(y.shape)
    # print(np.median(y, axis=0).astype(np.int64))
    ylist.append(np.mean(y, axis=0))
yarr= np.stack(ylist).astype(np.int64)
# yarr.sort()
# print(yarr)

ids = np.argsort(yarr.mean(axis=-1))
# print(yarr.shape, ids.shape)
yarr= yarr[ids].reshape(3,3,2)
# print(yarr.shape)
# exit()
iou = mybox_iou(yarr, labelarr)
print(iou.shape)
iou = iou.reshape(9, -1).transpose()
print(iou.shape)
thresh = 0.2
print(np.mean(np.max(iou, axis=-1)>thresh))
with open("F:\AI-046\Projects\yolov5_test_01\models\myyolov5s.yaml") as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
cocoanchor = hyp["anchors"]
print(cocoanchor)
cocoanchor = np.array(cocoanchor).reshape(3,3,2)
# cocoanchor = np.array([18,22,  53,30,  42,77,  120,84,  102,170,  146,282,  271,228,  372,382,  623,432]).reshape(3,3,2)
print(cocoanchor)
cocoiou = mybox_iou(cocoanchor, labelarr).reshape(9, -1).transpose()
print(np.mean(np.max(cocoiou, axis=-1)>thresh))
# print(labelarr[0])
# print(cocoiou[0])
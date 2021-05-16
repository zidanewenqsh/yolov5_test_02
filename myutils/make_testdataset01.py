'''
有问题,图片和标签不匹配,已经在yolov5_dents_01中修改
'''
#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        make_testdataset01.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/20 13:44
# @Description:      将yolov4的训集中取出部分作为验证集
# Function List:
# History:
#       <author>    <version>   <time>      <desc>
#       wen         ver0_1      2020/12/15  xxx
# ------------------------------------------------------------------
import os
import numpy as np
import torch
import cv2
import time
import shutil
from pathlib import Path
TESTNUM = 500
# basepath = Path(r"F:/FProjectsData/dents06/datas/Augments/saves/Aug_dents_ng_id20")
basepath = Path(r"F:\FProjectsData\woods05\datas\Auguments\saves\Aug_woods_ng_id40_3")
trainpath = basepath/"train"
assert trainpath.exists()
testpath = basepath/"test2"
if testpath.exists():
    shutil.rmtree(str(testpath))
testpath.mkdir(parents=True, exist_ok=True)
trainimgfiles = list(trainpath.glob("**/*.jpg"))
trainlabelfiles = list(trainpath.glob("**/*.txt"))
print(len(trainimgfiles), len(trainlabelfiles))
print("*"*100)
len_trainfiles = min(len(trainimgfiles), len(trainlabelfiles))
testimgdir = testpath/"images"
testlabeldir = testpath/"labels"
testimgdir.mkdir(parents=True, exist_ok=True)
testlabeldir.mkdir(parents=True,exist_ok=True)
ids = torch.randperm(len_trainfiles)
print(ids)
selectimgfiles = np.array(trainimgfiles)[ids][:min(len_trainfiles, TESTNUM)].tolist()
selectlabelfiles = np.array(trainlabelfiles)[ids][:min(len_trainfiles, TESTNUM)].tolist()
# exit()
print(selectimgfiles)
print(selectlabelfiles)
print(len(selectlabelfiles), len(selectlabelfiles))
for selectimgfile in selectimgfiles:
    imgname = selectimgfile.name
    testimgfile = testimgdir/imgname
    print(imgname)
    print(testimgfile)
    shutil.copy(selectimgfile, testimgfile)

for selectlabelfile in selectlabelfiles:
    labelname = selectlabelfile.name
    testlabelfile = testlabeldir/labelname
    print(labelname)
    print(testlabelfile)
    shutil.copy(selectlabelfile, testlabelfile)
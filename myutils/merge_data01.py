#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        merge_data01.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/19 15:10
# @Description:      将我的生成的四类数据和郝工造的四类数据进行混合
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
import shutil
from pathlib import Path
srcdir = Path(r"F:\FProjectsData\woods05\datas\Auguments\datas\lousha_0219")
dstdir = Path(r"F:\FProjectsData\woods05\datas\Auguments\saves\Aug_woods_ng_id41_1")
dstimgdir = dstdir/"images"
dstlabeldir = dstdir/"labels"
dstimgdir.mkdir(parents=True, exist_ok=True)
dstlabeldir.mkdir(parents=True, exist_ok=True)
# dstdir.mkdir(parents=True, exist_ok=True)
srcimgs = list(srcdir.glob("**/images/*.jpg"))
# srclabels = list(srcdir.glob("**/labels/*.txt"))
print(len(srcimgs))
for srcimg in srcimgs:
    print(srcimg)
    srclabel = srcimg.parent.parent/"labels"/f"{srcimg.stem}.txt"
    # print(srclabel)
    # print(srclabel.exists())
    if srcimg.exists() and srclabel.exists():
        shutil.copy(srcimg, dstimgdir)
        shutil.copy(srclabel, dstlabeldir)

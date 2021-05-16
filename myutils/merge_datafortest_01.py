#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        merge_datafortest_01.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/19 16:09
# @Description:      将我的生成的四类数据和郝工造的四类数据进行混合, 将郝工的数据每个分类各取一定数量的与idtest混合用作测试集
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

NUM = 20
basesrcdir = Path(r"F:\FProjectsData\woods05\datas\Auguments\datas\lousha_0219")
basedstdir = Path(r"F:\FProjectsData\woods05\datas\Auguments\saves\Aug_woods_ng_idtest_1")
dstimgdir = basedstdir/"images"
dstlabeldir = basedstdir/"labels"
dstimgdir.mkdir(parents=True, exist_ok=True)
dstlabeldir.mkdir(parents=True, exist_ok=True)
# dstdir.mkdir(parents=True, exist_ok=True)
srcimgdirs = list(basesrcdir.glob("**/images"))
print(srcimgdirs)
for srcimgdir in srcimgdirs:
    srcimgs = list(srcimgdir.glob("*.jpg"))
    idxs = torch.randperm(len(srcimgs))[:NUM].tolist()
    srcimgs = np.array(srcimgs)[idxs].tolist()
    print(len(srcimgs))
    print(idxs)
    for srcimg in srcimgs:
        print(srcimg)
        srclabel = srcimg.parent.parent/"labels"/f"{srcimg.stem}.txt"
        # print(srclabel)
        # print(srclabel.exists())
        if srcimg.exists() and srclabel.exists():
            shutil.copy(srcimg, dstimgdir)
            shutil.copy(srclabel, dstlabeldir)

# srclabels = list(srcdir.glob("**/labels/*.txt"))
# print(len(srcimgs))
# for srcimg in srcimgs:
#     print(srcimg)
#     srclabel = srcimg.parent.parent/"labels"/f"{srcimg.stem}.txt"
#     # print(srclabel)
#     # print(srclabel.exists())
#     if srcimg.exists() and srclabel.exists():
#         shutil.copy(srcimg, dstimgdir)
#         shutil.copy(srclabel, dstlabeldir)

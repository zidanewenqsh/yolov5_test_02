#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        mydetectall_01.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/20 8:33
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
import sys
import multiprocessing as mp
sys.path.append("/")
# p1 = Path("runs/train/woods/weights/last.pt")
# print(p1.exists())

cmdlist = [
    "python mydetect.py --device 0 --name test --weights runs/train/woods3/weights/last.pt"
]
os.system(cmdlist[0])
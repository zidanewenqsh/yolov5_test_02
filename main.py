#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        main.py.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/19 12:56
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
from tool.utils import record_log, get_time


cmddict = {
    0: f"python mytrain.py --epochs 20 --batch-size 16 --data data/woodstest.yaml --name test --time {get_time(0)}",
    1: f"python mytrain.py --epochs 20 --batch-size 16 --data data/woodstest.yaml --name test --device 1 --check-point 3 --time {get_time(0)}",
    2: f"python mytrain.py --epochs 500 --batch-size 16 --data data/woods.yaml --name woods --device 2 --check-point 5 --time {get_time(0)}",
    3: f"python mytrain.py --epochs 1000 --batch-size 16 --data data/woods.yaml --name woods --device 3 --check-point 5 --time {get_time(0)}",
    4: f"python mytrain.py --epochs 1000 --batch-size 16 --data data/woods.yaml --name woods --device 1 --check-point 5 --warmup-epoch 100 --time {get_time(0)}",
    5: f"python mytrain.py --epochs 500 --batch-size 16 --data data/woods.yaml --name test --device 1 --check-point 5 --warmup-epoch 100 --time {get_time(0)}",
    6: f"python train2.py --epochs 500 --batch-size 16 --data data/woods.yaml --name woodstrain --device 2  --time {get_time(0)}",
    7: f"python mytrain.py --epochs 500 --batch-size 16 --data data/woods.yaml --name woodsmytrain --device 3 --check-point 1 --warmup-epoch 1000 --time {get_time(0)}",

}
for k, v in cmddict.items():
    print(k, end=" ")
    print(v)
# exit()
index = int(input(f"len of cmddict: {len(cmddict)}\ninput index: "))
if index>=len(cmddict):
    exit(100)
cmd = cmddict[int(index)]
# with open("record.txt", 'a') as f:
#     print(cmd, file=f)
record_log(cmd)
os.system(cmd)
#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        main.py
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
import multiprocessing as mp
from tool.utils import record_log, get_time, loginfo
import logging


def logger(content: str, logfile=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    # 第二步，创建一个handler，用于写入日志文件
    if logfile:
        fh = logging.FileHandler(logfile, mode='a')
    else:
        fh = logging.StreamHandler()

    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    # 日志
    logger.info(content)
    logger.removeHandler(fh)
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置

# cmddict = {
#     0: f"python mytrain.py --epochs 20 --batch-size 16 --data data/woodstest.yaml --name test --time {get_time(0)}",
#     1: f"python mytrain.py --epochs 20 --batch-size 16 --data data/woodstest.yaml --name test --device 1 --check-point 3 --time {get_time(0)}",
# }
cmdalllist = [
    f"python mytrain.py --epochs 500 --batch-size 16 --data data/oca.yaml --name oca --device 1 --check-point 1 --time {get_time(0)}",
    f"python mytrain.py --epochs 500 --batch-size 16 --data data/woods.yaml --name woods --device 2 --check-point 1 --time {get_time(0)}",
    f"python mytrain.py --epochs 50 --batch-size 16 --data data/oca.yaml --name test --device 0 --check-point 1 --time {get_time(0)}",
    f"python mytrain.py --epochs 50 --batch-size 32 --data data/woodstest.yaml --name test --device 1 --check-point 3 --time {get_time(0)}",
    f"python mytrain.py --epochs 50 --batch-size 16 --data data/woodstest.yaml --name test --device 2 --check-point 3 --time {get_time(0)}",
    f"python mytrain.py --epochs 50 --batch-size 16 --data data/woodstest.yaml --name test --device 2 --check-point 3 --time {get_time(0)}",
    f"python mytrain.py --epochs 50 --batch-size 8 --data data/woodstest.yaml --name test --device 1 --check-point 3 --time {get_time(0)}",
]
cmdlist = cmdalllist[:2]
def run(cmd):
    loginfo(cmd, "record.txt")
    loginfo(cmd)
    # record_log(cmd)
    os.system(cmd)
if __name__ == '__main__':
    plist = []
    for cmd in cmdlist:
        p = mp.Process(target=run, args=(cmd,))
        plist.append(p)
        print(p.name)
        p.start()
        time.sleep(5)
    for p in plist:
        p.join()
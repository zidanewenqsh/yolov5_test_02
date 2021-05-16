#!/usr/bin/python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        augmentok_class_yolo_dents_v7.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/1/29 13:19
# @Description:      Main Function:    xxx
# Function List:     hello() -- print helloworld
# History:
#       <author>    <version>   <time>      <desc>
#       wen         ver0_1      2020/12/15  xxx
# ------------------------------------------------------------------
import sys
import os
import numpy as np
import torch
import cv2
import time
import random
import math
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import threading

FILEPATH = os.path.abspath(os.path.dirname(__file__))
CODESPATH = os.path.dirname(FILEPATH)
BASEPATH = os.path.dirname(CODESPATH)
if CODESPATH not in sys.path:
    sys.path.append(r"D:\AI-046\Ubuntu\Projects\dents06\dents06codes")

from tool.utils import makedir, removepath, getpath
# from tool2.utils2 import makedir, get_gaussian_kernel, removepath
from tool2.algorithms_converse import img_converse1
import shutil

# BLURFLAG = 2 # 1 为高斯滤波， 2为中值滤波
MULTITHREAD = 0
if MULTITHREAD:
    TRAINNUM = 5000
    TESTNUM = 100
else:
    TRAINNUM = 10
    TESTNUM = 10

CHECKNUM = 100

BLACKTHRESH = 10

NEWGEN = True
ISTEST = False
THREADNUM = 16
LOCK = threading.Lock()
SAVE = True
INDEX = 0
MASKTHRESH = 10
DILATE_SIZE = 10
IMGHEIGHT = 3500
IMGWIDTH = 2048
SAVEIMGSIZE = 640
MASKSIZE = 512
savebase_dir = getpath(r"F:\FProjectsData\dents06\datas\Augments\saves")
database_dir = getpath(r"F:\FProjectsData\dents06\datas\Augments\datas")

# augdictsavetrain_file = os.path.join(savebase_dir, "augtrain_dents_ng_id21.dict")
# augdictsavetest_file = "../saves/augtest_dents_ng_id21.dict1"

img_dir = getpath(r"F:\FProjectsData\dents06\datas\Augments\datas\OK-20201229")
remainimg_dir = getpath(r"F:\FProjectsData\dents06\datas\Augments\datas\OK-20201229-save-pass_v1229_t0202")
# yahen-region_v1229_t0202
roiimgbase_dir = os.path.join(database_dir, "yahen-region_v1229_t0202")
# roiimgbase_dir = os.path.join(database_dir, "oca_ng_hao_01")
anchordict_path = r"F:\FProjectsData\dents06\datas\Augments\datas\OK-20201229.dict"
save_dir = os.path.join(savebase_dir, "Aug_dents_ng_id21")
print(save_dir)

makedir(save_dir)

# ngsave_dir = os.path.join(savebase_dir, "Aug_dents_ng_id21", "ng")
#
# makedir(ngsave_dir)
'''
v1 for dents bubble
id 1
id 2
id 3 浅坑中间填充椭圆
id 4 加了大小调节和亮度调节
v3: 气泡凹坑自己制作的掩码
v3_2: 膨胀
    id6: 气泡小区域 
    id7: 气泡大区域
    id8: 增大ng数据量 901 + 518 共 10000
    id 09: like id 08 共10000, size changed
    id 10 pit minsize increaed, img size 1024, mask size 512
    id 11 use pit data treaed by HAO, obvious pit reserved
    id 12 size 2048
    id 13 ok img flip vertical. imgsize 2048
    id 14 pit min size decreased, imgsize 2048, about 17 ng fillimg added both in train and test
    id 15 pit min size decreased, other as 14
v4:v3_2
    id 16 凹坑区域改进, ng图只起权重作用, 将ng图除以掩码我的平均值（或中值），与原ok区域相乘。直接作为原图区域，或是可以按原来的膨胀掩码修改原图，可以模糊。 
v5:v4:
    for yolov5 qipao aokeng
    id 17
v6:v5
    for yolov5 dents
    id 18
v7:v6
    for yolov5 dents fourier模板
    id 19
    id 20 减少对比度，减少高度比较大的比例，仿射变换放大取中间
v7_2:
    在v7的基础上造一些浅的
    id 21

'''


def gen_fourier(srcsize=1000, dstwidth=300, dstheight=500, n=2, thresh=1e-2, twodim=False):
    assert n > 1, "n must more than one"
    while True:
        x = np.linspace(0, 2 * np.pi, srcsize)
        y = np.zeros_like(x)
        for i in range(1, n):
            a1, a2 = np.random.randint(90, 110, (2,)) / (100 * i)
            # print(f"a1:{a1}, a2:{a2}")
            w1, w2 = i * np.random.randint(50, 200, (2,)) / 100
            b1, b2 = 2 * np.pi * np.random.randint(-50, 50, (2,)) / 100
            y += a1 * np.cos(w1 * n * x + b1) + a2 * np.sin(w2 * n * x + b2)

        ids = np.nonzero(np.abs(y) < thresh)[0]
        if ids.shape[0] < 3:
            continue
        start, end = ids[0], ids[-1]
        if (end - start) > 100:
            y_ = y[start:end]
            if np.max(y_) * np.min(y_) < 0:
                break
    # x_ = x[start:end]
    y_ = y[start:end]
    a3 = np.random.randint(20, 50) / 100
    y_ = a3 * 0.5 * y_ / np.max(np.abs(y_))
    y_ = cv2.resize(y_.reshape(-1, 1), (1, dstheight))
    # print("maxmin", np.max(y_), np.min(y_))
    if twodim:
        y_ = y_.repeat(dstwidth, 1)
        return y_
    else:
        return y_.flatten()


def gen_mask(h_, w_):
    '''
    生成偏圆形的mask
    :param h_:
    :param w_:
    :return:
    '''
    # 固定大小
    h = 100
    w = h
    img2 = np.zeros((h, w), np.uint8)
    color = (255,)
    # 画圆
    center = (w // 2, h // 2)
    # try:
    ## 参数调小
    radius = np.random.randint(min(w, h) // 4, min(w, h) // 3)

    cv2.circle(img2, center, radius, color, -1)
    # 边界为0
    margin = 5
    # 画椭圆
    # 极坐标取点
    n = np.random.randint(10, 50)
    thetas = np.random.uniform(0, 2 * np.pi, size=(n, 1))
    for theta in thetas:
        angle = np.random.randint(0, 361)
        startAngle = np.random.randint(0, 180)
        endAngle = np.random.randint(180, 361)
        cx = radius * np.cos(theta) + w // 2
        cy = radius * np.sin(theta) + h // 2

        center = tuple(np.array([cx[0], cy[0]], 'i4').tolist())

        axex_x = np.random.randint(w // 32, w // 4)
        axex_y = np.random.randint(h // 32, h // 4)

        cv2.ellipse(img2, center, (axex_x, axex_y), angle, startAngle, endAngle, color, -1)

    # img2 = img_converse(img2)
    # 开操作
    s = cv2.getStructuringElement(cv2.MORPH_OPEN, (7, 7))
    # print(s)
    img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, s)
    # 去边
    img2[:margin, :] = 0
    img2[h - margin:, :] = 0
    img2[:, :margin] = 0
    img2[:, w - margin:] = 0
    # 只取大于指定阈值部分
    img2[img2 < 255] = 0
    img2 = cv2.resize(img2, (w_, h_))

    return img2


class AugNeg:
    '''
    aug for ok
    range(50,200), mean 120, std 50
    add linear conversion

    '''

    def __init__(self, savedir=save_dir):
        self.savedir = savedir
        self.index = 0
        self.trainlabelpath = os.path.join(save_dir, "trainlabel_id21.dict")
        self.testlabelpath = os.path.join(save_dir, "testlabel_id21.dict")
        self.anchordict = torch.load(anchordict_path)
        # print("self.anchordict",self.anchordict.keys(),len(self.anchordict))
        # print(self.anchordict.keys())
        # exit(1804)
        # # self.augdictsavefile = augdictsavefile
        # # self.savedir = savedir
        # # self.augdict = {}
        # # self.index = 0
        # self.ngid = 0
        # # 选择基文件夹，不同分类下有img1代表不模糊， img2代表模糊
        self.roiimgdirs = os.listdir(roiimgbase_dir)
        self.probablitylist = []
        # 文件夹名的最后是概率
        for dirname in self.roiimgdirs:
            self.probablitylist.append(float(dirname.split(".")[0].split("-")[-1]) / 100)
        # # self.save = False

    def get_ng(self, roiimgdir_):
        '''
         生成ng区域
        :param roiimgdir:
        :return:
        '''
        # assert roiimgdir != None
        # 选择生成ng的文件夹。默认 小黑缺陷

        # exit()
        # roiimgdir = roiimgdir_
        self.flag1 = True if roiimgdir_.startswith("1") else False  # 1 for 坑
        # self.flag2 = True if roiimgdir_.startswith("2") else False  # 2 for 气泡

        cls = int(roiimgdir_[0])

        if self.flag1:
            roiimgdir = os.path.join(roiimgbase_dir, roiimgdir_, "img-1")
        else:
            roiimgdir = os.path.join(roiimgbase_dir, roiimgdir_, "img")

        imgnamelist = os.listdir(roiimgdir)

        # masknamelist = os.listdir(roimaskdir)
        len_imgnamelist = len(imgnamelist)

        # 取ng图的索引
        i3 = np.random.randint(0, len_imgnamelist)
        # 选择ng图
        imgname = imgnamelist[i3]
        #     imgname = f"{imgname}.jpg"
        imgpath = os.path.join(roiimgdir, imgname)

        # 打开图片
        img_ng = cv2.imread(imgpath, 0)
        return img_ng, imgname, cls

    def merge(self, srcimg, imgid, savedir, labelsavedir, labelpath):
        # print(f"imgid:{imgid}")
        # exit(0)
        global INDEX
        if MULTITHREAD:
            LOCK.acquire()
        self.index = INDEX
        INDEX += 1
        if (INDEX % 10 == 0):
            print(f"{INDEX}", end=" ")
            # print(f"=", end=" ")
            if INDEX % 100 == 0:
                print()
        if MULTITHREAD:
            LOCK.release()
        height_src, width_src = srcimg.shape
        # print(f"imgid----{imgid}")
        _x1, _y1, _x2, _y2 = self.anchordict[f"{imgid}.jpg"]

        #  按概率选择文件夹
        roiimgdir_N = torch.multinomial(torch.Tensor(self.probablitylist), 1).item()
        roiimgdir = self.roiimgdirs[roiimgdir_N]
        # cls = roiimgdir[0]
        # 设置选择模糊文件的概率40%
        # isblur = np.random.randint(10) > 5
        # 保存的文件名
        savename = f"{roiimgdir[0]}_{imgid}_{self.index}.jpg"  #

        labeldict = {}
        labeldict[savename] = []

        # 复制原ok文件
        dstimg = srcimg[_y1:_y2, _x1:_x2]
        # dstimg = srcimg[:, _x1:_x2]
        # dstimg = srcimg[_y1:_y2, :]
        if 0:
            r = 0.3
            srcimg_ = cv2.resize(srcimg, (0, 0), fx=r, fy=r)
            dstimg_ = cv2.resize(dstimg, (0, 0), fx=r, fy=r)
            print(srcimg.shape, dstimg.shape)
            cv2.imshow("srcimg", srcimg_)
            cv2.imshow("dstimg", dstimg_)
            cv2.waitKey(0)
            exit(1406)
        # 空白掩码
        # srcmask = np.zeros_like(srcimg, dtype=np.uint8)
        # dstmask = np.zeros_like(dstimg, dtype=np.uint8)
        # 图片宽高
        height, width = dstimg.shape
        # ngimg, imgname, cls = self.get_ng(roiimgdir)
        # h_ratio = np.random.randint(20, 200)/100
        w_ratio = np.random.randint(90, 100) / 100
        if True:
            mu = torch.ones(1) * 60
            sigma = torch.ones(1) * 60
            min_hratio = 5
            max_hratio = 200
            while True:
                h_ratio = torch.normal(mu, sigma).int().item()
                if h_ratio > min_hratio and h_ratio < max_hratio:
                    h_ratio /= 100
                    break

        # print(h_ratio)
        # exit()
        # print(imgname)
        # x1 = np.random.randint(5, 50)
        w = int((_x2 - _x1) * w_ratio)
        _cx = (_x2 - _x1) // 2
        start_y = 10
        h = min(int(w * h_ratio), height - start_y - 10)
        # ngimg = cv2.resize(ngimg, (w,h))
        # print(w, h)
        # start_y = 10
        # print(height-h)
        y_up = np.random.randint(start_y, height - h)
        # print(y_up)
        # y_down = y_up + h
        # print(y_down, _y2)
        # print(ngimg.shape)

        # exit(1947)
        # # flip according to i4
        # if i4 > 0:
        #     ngimg = ngimg[::-1, ::-1]
        #     maskimg = maskimg[::-1, ::-1]
        # # 去掉掩码特别小的
        # if (np.sum(maskimg > 10) < 100):
        #     continue
        #
        # # self.ngid += 1
        #
        # # print(ngimg.shape)
        # h, w = ngimg.shape
        # # print(max_ng, w, h)
        # # 坐标换算
        # x1 = x1_id_ * max_ng + np.random.randint(0, max(max_ng - w, 1))
        # y1 = y1_id_ * max_ng + np.random.randint(0, max(max_ng - h, 1))
        # x1 = np.random.randint(5,50)
        x1 = _cx - w // 2
        y1 = y_up
        x2 = x1 + w
        y2 = y1 + h
        cx = x1 + w // 2 + _x1
        cy = y1 + h // 2 + _y1
        # print(f"x1:{x1,y1,x2,y2}")
        # print(ngimg[0:h//10].shape, ngimg[-h//10+1:].shape)
        # ngimg_out = np.concatenate((ngimg[0:h//10], ngimg[-h//10+1:]))
        # print(ngimg_out.shape)

        # mask_out_compt_median = np.mean(ngimg_out)
        # print(mask_out_compt_median)
        roiimg = dstimg[y1:y2, x1:x2]
        # print(roiimg.shape,ngimg.shape)

        # ngimg_normal = ngimg.astype(float) / mask_out_compt_median
        # n 0.2-0.55-0.9-
        # print(roiimg.shape, h,w,height,width, h_ratio)
        ngimg_normal = gen_fourier(srcsize=1000, dstwidth=w, dstheight=h, n=(2 + int((h_ratio - 0.2) / 0.6)),
                                   thresh=1e-2, twodim=True)
        ngimg_normal = img_converse1(ngimg_normal)
        ngimg_normal = cv2.resize(ngimg_normal, (0,0), fx=2, fy=1)[:,(w//2):(w//2+w)]
        # if 1:
        #     print(ngimg_normal.shape, w, h)
        #     exit(958)
        # [:,(w//2):(w//2+w)]
        ngimg_normal += 1

        roiimg_ = roiimg.astype(float) * ngimg_normal
        roiimg_ = roiimg_.clip(0, 255).astype(np.uint8)
        dstimg[y1:y2, x1:x2] = roiimg_

        cx_ = float(cx / width_src)
        cy_ = float(cy / height_src)
        w_ = float(w / width_src)
        h_ = float(h / height_src)
        cls = 2
        labeldict[savename].append((f"{cls} {cx_} {cy_} {w_} {h_}"))

        if 1:
            if MULTITHREAD:
                LOCK.acquire()
            labeldict_ = torch.load(labelpath) if os.path.exists(labelpath) else {}
            labeldict_.update(labeldict)
            torch.save(labeldict_, labelpath)
            if MULTITHREAD:
                LOCK.release()
            savepath = os.path.join(savedir, savename)
            labelsavepath = os.path.join(labelsavedir, f"{savename.split('.')[0]}.txt")
            srcimg = srcimg.clip(0, 255).astype(np.uint8)

            # dstmask = dstmask.astype(np.uint8)
            # dstmask = np.clip(dstmask, 0, 255)
            srcimg = cv2.resize(srcimg, (int(width_src * SAVEIMGSIZE / height_src), SAVEIMGSIZE))
            # dstmask = cv2.resize(dstmask, (MASKSIZE, MASKSIZE))
            cv2.imwrite(savepath, srcimg)
            with open(labelsavepath, 'w') as f:
                for line in labeldict[savename]:
                    print(line, file=f)
            # cv2.imwrite(masksavepath, dstmask)

        self.index += 1

    def generate_train(self):
        print("gen train start")
        global INDEX
        imgdir = img_dir
        # imgdir_list = os.listdir(imgdir)
        # imgdir_listlen = len(imgdir_list)
        remainimgdir = remainimg_dir
        imgdir_list = os.listdir(remainimgdir)
        imgdir_listlen = len(imgdir_list)

        # trainsave_dir = os.path.join(self.savedir, "train")
        # masksave_dir = os.path.join(self.savedir, "trainmask")
        trainsave_dir = os.path.join(self.savedir, "train", "images")
        labelsave_dir = os.path.join(self.savedir, "train", "labels")

        if NEWGEN:
            removepath(trainsave_dir)
            removepath(labelsave_dir)
            # print("exist:",os.path.exists(trainsave_dir), os.path.exists(masksave_dir))

        makedir(trainsave_dir)
        makedir(labelsave_dir)
        # print(f"{len(trainsave_dir)} {len(masksave_dir)}")

        INDEX = min(len(os.listdir(trainsave_dir)), len(os.listdir(labelsave_dir)))
        # print(f"index:{INDEX}")
        tids = []
        while (INDEX < TRAINNUM):
            # print(f"index:{INDEX}, trainnum:{TRAINNUM}")
            index = np.random.randint(0, imgdir_listlen)
            imgname = imgdir_list[index]
            imgid = imgname.split(".")[0]

            imgpath = os.path.join(imgdir, imgname)
            if imgname not in self.anchordict.keys():
                continue
            # print(imgpath)
            try:
                srcimg = cv2.imread(imgpath, 0)
            except Exception as e:
                print(imgpath)
                exit(0)
            # for i in range(THREADNUM):
            if MULTITHREAD:
                tid = threading.Thread(target=self.merge,
                                       args=[srcimg, imgid, trainsave_dir, labelsave_dir, self.trainlabelpath])
                tids.append(tid)
                tid.start()
                if len(tids) >= THREADNUM:
                    tids = []
                    for tid in tids:
                        tid.join()
            else:
                self.merge(srcimg, imgid, trainsave_dir, labelsave_dir, self.trainlabelpath)

        print()
        time.sleep(0.1)

    def generate_test(self):
        # imgdir = img_dir
        # remainimgdir = remainimg_dir
        # imgdir_list = os.listdir(remainimgdir)
        # imgdir_listlen = len(imgdir_list)
        print("gen test start")
        global INDEX
        imgdir = img_dir
        remainimgdir = remainimg_dir
        imgdir_list = os.listdir(remainimgdir)
        imgdir_listlen = len(imgdir_list)
        # testsave_dir = os.path.join(self.savedir, "test")
        # testmasksave_dir = os.path.join(self.savedir, "testmask")
        testsave_dir = os.path.join(self.savedir, "test", "testimages")
        testlabelsave_dir = os.path.join(self.savedir, "test", "testlabels")
        if NEWGEN:
            removepath(testsave_dir)
            removepath(testlabelsave_dir)

        makedir(testsave_dir)
        makedir(testlabelsave_dir)
        INDEX = min(len(os.listdir(testsave_dir)), len(os.listdir(testsave_dir)))
        tids = []
        while (INDEX < TESTNUM):
            index = np.random.randint(0, imgdir_listlen)
            imgname = imgdir_list[index]
            imgid = imgname.split(".")[0]

            if imgname not in self.anchordict.keys():
                continue
            imgpath = os.path.join(imgdir, imgname)

            srcimg = cv2.imread(imgpath, 0)
            # print(f"srcimg:{srcimg.shape}")
            # exit()
            if MULTITHREAD:
                tid = threading.Thread(target=self.merge,
                                       args=[srcimg, imgid, testsave_dir, testlabelsave_dir, self.testlabelpath])
                tids.append(tid)
                tid.start()
                if len(tids) >= THREADNUM:
                    for tid in tids:
                        tid.join()
                    tids = []
            else:
                self.merge(srcimg, imgid, testsave_dir, testlabelsave_dir, self.testlabelpath)

        print()
        time.sleep(0.1)

    def generate(self):
        self.generate_train()
        self.generate_test()

    #     print("sys.argv", len(sys.argv))
    #     if (len(sys.argv)>1):
    #         self.generate_test()
    #     else:
    #         self.generate_train

    def load_label(self):
        traindict = torch.load(self.trainlabelpath)
        # testdict = torch.load(self.testlabelpath)
        print(f"dictlen: {len(traindict)}")
        print(traindict)

        train_dir = os.path.join(self.savedir,"train", "images")
        save_dir = os.path.join(self.savedir, "dictsave")
        removepath(save_dir)
        time.sleep(1)
        makedir(save_dir)

        imglist = os.listdir(train_dir)
        print(imglist)

        ids = torch.randperm(len(imglist)).numpy()[:CHECKNUM]
        print(ids)
        imglist = list(np.array(imglist)[ids])
        # tq = tqdm(imglist)
        for i, imgname in enumerate(imglist):
            imgfile = os.path.join(train_dir, imgname)
            img = cv2.imread(imgfile)
            height, width = img.shape[:2]
            savepath = os.path.join(save_dir, imgname)

            # print(savepath)
            if imgname not in traindict.keys():
                print(f"{imgname} not in keys")
                continue
            for coord in traindict[imgname]:
                # print(len(coord))
                # cls, cx, cy, w, h = [int(float(x)*IMGSIZE) for x in coord.split(" ")]
                cls, cx, cy, w, h = coord.split(" ")
                cx = int(float(cx) * width)
                w = int(float(w) * width)
                cy = int(float(cy) * height)
                h = int(float(h) * height)

                x1 = cx - w // 2
                y1 = cy - h // 2
                x2 = x1 + w
                y2 = y1 + h
                print(cls, cx, cy, w, h)
                # print(x1,x2,y1,y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.imwrite(savepath, img)
                # cv2.imshow(imgname, img)
                # cv2.waitKey(0)
            # if i > 10:
            #     break


if __name__ == '__main__':
    an = AugNeg()
    t1 = time.time()
    an.generate()
    an.load_label()
    # an.checkngimg()
    t2 = time.time()
    print(f"time: {t2 - t1}")
    # an.checkngimg()

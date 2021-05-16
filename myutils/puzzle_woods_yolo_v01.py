#!/e/Soft/Python/Python38/python
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        puzzle_woods_yolo_v01.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/22 16:54
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
import threading

# basedir = Path("F:/FProjectsData/woods05/datas/Auguments/saves/Aug_woods_ng_id41")
basedir = Path("F:/FProjectsData/woods05/datas/Auguments/saves/Aug_woods_ng_id41")
# traindir = basedir/"train"
# testdir = basedir/"test"
savebasedir = basedir/"puzzle"
np.set_printoptions(suppress = True)
# savetraindir = savebasedir/"train"
# savetestdir = savebasedir/"test"
modes = {"train":10000, "test":1000}
# TRAINNUM = 100
# TESTNUM = 100
INDEX=0
def readlabel(labelpath, width, height):
    # print(labelpath)
    labellist = []
    with open(labelpath) as f:
        content = f.readlines()
        for c in content:
            # print(c)
            # exit()
            labellist.append([float(x) for x in c.strip().split(" ")])
        # print(labellist)
    labelarr = np.array(labellist)
    labelarr[:, 1::2]*=width
    labelarr[:, 2::2]*=height
    # print(labelarr)
            # print(c.decode('utf-8'))
    return labelarr

def handler(listlen, stemlist, imgsrcdir, labelsrcdir, imgdstdir, labeldstdir, mode, INDEX):
    # print(num)
    # exit()
    indices = torch.randperm(listlen)[:4].tolist()
    stemlist_ = np.array(stemlist)[indices].tolist()
    # print(stemlist_)
    srcpaths = [imgsrcdir / f"{x}.jpg" for x in stemlist_]
    labelspaths = [labelsrcdir / f"{x}.txt" for x in stemlist_]
    # print(labelspaths[0].exists())
    # exit()
    imglist = [cv2.imread(str(x), 0).T for x in srcpaths]
    imgarr = np.concatenate(imglist, 0).T
    height, width = imgarr.shape
    width /= 4
    # print(imgarr.shape)
    imgsavename = f"{mode}_{INDEX}.jpg"
    labelsavename = f"{mode}_{INDEX}.txt"
    print(imgsavename)
    imgsavepath = imgdstdir / imgsavename
    labelsavepath = labeldstdir / labelsavename
    # print(imgsavepath)
    cv2.imwrite(str(imgsavepath), imgarr)
    labelcontents = [readlabel(x, width, height) for x in labelspaths]
    # print(len(labelcontents))
    for i, labelcontent in enumerate(labelcontents):
        labelcontent[:, 1] += i * width
        labelcontent[:, 1:] /= height
    # print(labelcontents)
    labelcontents = np.concatenate(labelcontents, 0)
    # print(labelcontents)
    # print(labelcontents.shape)
    with open(labelsavepath, 'w') as f:
        for x in labelcontents:
            cls, cx, cy, w, h = x
            cls = int(cls)
            print(f"{cls} {cx} {cy} {w} {h}", file=f)


def trans():
    global INDEX
    for mode in modes.keys():
        INDEX = 0
        print(f"mode:{mode}")
        srcdir = basedir/mode
        dstdir = savebasedir/mode
        imgsrcdir = srcdir/"images"
        labelsrcdir = srcdir/"labels"
        imgdstdir = dstdir/"images"
        labeldstdir = dstdir/"labels"
        imgdstdir.mkdir(parents=True, exist_ok=True)
        labeldstdir.mkdir(parents=True, exist_ok=True)
        srcimglist = list(srcdir.glob("**/*.jpg"))
        # srclabellist = list(srcdir.glob("**/*.txt"))
        stemlist = [x.stem for x in srcimglist]
        listlen = len(stemlist)
        # num = len(stemlist) # 17971
        # print(stemlist)
        # print(len(srcimglist), srcdir.exists(), srcdir)
        # indiceslist = []
        # for i in range(4):
        #     indices = torch.randperm(NUM).tolist()
        #     indiceslist.append(indices)
        # print(indiceslist)
        # indicesarr = np.array(indiceslist).T
        # print(indicesarr.shape)
        num = modes[mode]

        tdlist = []
        while INDEX<num:
            td = threading.Thread(target=handler, args=[listlen, stemlist, imgsrcdir, labelsrcdir, imgdstdir, labeldstdir, mode, INDEX])
            tdlist.append(td)
            td.start()
            if INDEX %10==0:
                for td_ in tdlist:
                    td_.join()
                tdlist = []
            INDEX += 1
            # print(labelcontents)
            # for x in labelspaths:
            #     print(x.exists())
            #     lablecontent = readlabel(x)
                # print(lablecontent)

            # break
            # imgarr_ = cv2.resize(imgarr, (640,640))
            # cv2.imshow("test", imgarr_)
            # cv2.waitKey(0)
            # print(srcpaths[0].exists(), srcpaths[0])
            # for img in imglist:
            #     print(img.shape)
            # print(indices)
            # print(indices)
            # stemarr = np.array(stemlist)[indices]
            # print(stemarr)
        # print(srcimglist)


def check():
    checksavedir = savebasedir/"check"
    checksavedir.mkdir(parents=True, exist_ok=True)
    imgdir = savebasedir/"train" / "images"
    labeldir = savebasedir/"train" / "labels"
    print(imgdir.exists(), labeldir.exists())
    imglist = list(imgdir.glob("*.jpg"))
    labellist = list(labeldir.glob("*.txt"))
    print(len(imglist))
    imglistlen = len(imglist)
    indices = torch.randperm(imglistlen)[:min(50, imglistlen)].tolist()
    print(indices)
    imgarr = np.array(imglist)[indices]
    labelarr = np.array(labellist)[indices]
    # print(imgarr)
    # print(labelarr)

    for i in range(imgarr.shape[0]):
        print(i)
        imgpath = imgarr[i]
        imgname = imgpath.name
        savepath = checksavedir/imgname
        print(savepath)

        labelpath = labelarr[i]
        img = cv2.imread(str(imgpath), 0)
        height, width = img.shape
        with open(labelpath) as f:
            for line in f.readlines():
                # print(line)
                infolist = [float(x) for x in line.strip().split(" ")]
                infoarr = np.array(infolist)
                infoarr[1::2] *= width
                infoarr[2::2] *= height
                infoarr = infoarr.astype(np.int32)
                cls, cx, cy, w, h = infoarr
                x1 = cx - w//2
                y1 = cy - h//2
                x2 = x1+w
                y2 = y1+h
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                # print(pt1, pt2)
                color = (250,)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(img, pt1, pt2, color)
                cv2.putText(img, f"{cls}", (x1, y1), font, 1, color)
        cv2.imwrite(str(savepath), img)
        # cv2.imshow("test", img)
        # cv2.waitKey()


                # print(cls, cx, cy, w, h)
                # cls = int(cls)
                # cx *= width
                # cy *= height
                # w *= width
                # h *= height
                # x1 = int(cx - w//2)
                # y1 = int(cy - h //2)

if __name__ == '__main__':
    trans()
    check()
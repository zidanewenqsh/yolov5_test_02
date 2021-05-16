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
from pathlib import Path
from tool.utils import makedir, getpath, getpath, removepath

MULTITHREAD = 1
if MULTITHREAD:
    TRAINNUM = 10000
    TESTNUM = 0
else:
    TRAINNUM = 100
    TESTNUM = 0
FILEPATH = os.path.abspath(os.path.dirname(__file__))
CODESPATH = os.path.dirname(FILEPATH)
# BASEPATH = os.path.dirname(CODESPATH)
CODESPATH = getpath(CODESPATH)
print(CODESPATH)
if CODESPATH not in sys.path:
    sys.path.append(CODESPATH)

# from tool2.algorithms_converse import img_converse2
# import shutil

# BLURFLAG = 2 # 1 为高斯滤波， 2为中值滤波
BLACKTHRESH = 20  # 防止生成在木材之外的区域
# TRAINNUM = 10000
# TESTNUM = 1000
NEWGEN = True
ISTEST = False
THREADNUM = 16
IMGSIZE = 1024
HEIGHTSIZE = 640*4
WIDTHSIZE = 640
MASKSIZE = 512
MASKTHRESH = 10
DILATE_SIZE = 10
CHECKNUM = 100
LOCK = threading.Lock()
# LOCK = threading.Lock()
SAVE = True
INDEX = 0
savebase_dir = getpath(r"F:\FProjectsData\woods05\datas\Auguments\saves")
database_dir = getpath(r"F:\FProjectsData\woods05\datas\Auguments\datas")

# 背景ok图路径
# img_dir = getpath(r"F:\FProjectsData\woods05\datas\Auguments\original\SPLIT_OK-212815-all") # 均衡化
# img_dir = getpath(os.path.join(database_dir, "SPLIT_20201219-DL-Train")) # 原图
# img_dir = getpath(os.path.join(database_dir, "SPLIT_OK-212815-all")) # 均衡化
img_dir = getpath(r"F:\FProjectsData\woods05\datas\Auguments\original\20201219-affined")  # 仿射化

print(os.path.exists(img_dir))

roiimgbase_dir = os.path.join(database_dir, "CLS_ROI_Defects8_20201119")
# roiimgbase_dir = os.path.join(database_dir, "SPLIT_20201219-DL-Train")
# F:\FProjectsData\woods05\datas\Auguments\datas\SPLIT_20201219-DL-Train
save_dir = os.path.join(savebase_dir, "Aug_woods_ng_id41")

makedir(save_dir)

'''
id6 亮度范围75-90%
id7 亮度范围80-95%
v4:id8 过拟合20201009，只把原图贴上
v5: id9处理鼓包
v6: id10处理扩大版roi, 尺寸80-220， 中心150，取消长宽比限制
id 11 掩码做小一点，灰度经郝工提前均衡化
v7: 
id 12 掩码均衡化边缘扩大到5个像素，将ng区域灰度值调亮一点85-95%, 大小60-240，中心150
id 13 较大修改，尺寸变小，掩码变形
v8: 保存掩码
id 16 保存掩码，便于Unet训练
id 17 继续减小缺陷大小，亮度范围50-95%
id 18 小的灰缺陷深度增加，增加大的浅缺陷的尺寸
v9: 多分类
id 19 分三类：亮白，浅灰，黑
v10: 缺陷分割掩码
id 20  N=1无多ng融合
v11: 增加模糊原始图片
id 21 N=1无多ng融合
id 22 有多ng融合
v12: 最终图片进行模糊，增加缺陷个数，专为unet设计，兼顾Yolo
id 23 gaussian 5 blur
id 24 median 5 blur
id 25 ok图增加到5000多张，并用Augment增强到8192张。
id 26 ok图经过筛选后全部收集，没有用Augment 增强
id 27 最终图片去掉模糊，模糊操作改为之后进行,增加大而亮的缺陷，N = 1
v13:大修
id 28    gen_mask换新
    分类3浅坑改为用类椒盐算法从OK图中生成
id 29 修改了椒盐算法
    加了按比例整体变亮和变暗，对掩码大小有限制
v14 多线程    
id 30 加入多线程，去掉isblur
v15 直接调整亮度生成ng
id 30 31
v16 合并v14和v15的处理方法
    id 32
        直接resize到512
        增加掩码threshold
        修改black_threshold为20
        flags3
v17:处理的Ng图为自己用精灵标注助手标注的
    id 33 ng区域大小改变，部分最小值改小
    id 34 ok区域均衡化 size 2048
    id 35 ok区域均衡化 size 1024
v18:修改Ng区域处理方法，将ng区域设为权重，参考dents06/dents06codes/datatreat/augment/augmentok_class_dentes_v4.py
    id 36 ok图采用均衡化过的图片
    id 37 ok图采用均衡化图片，（粉尘斑加模糊,no）
        粉尘斑直接用原图，且大小有修改，与原图合并采用加权和
        取消对原图和掩码作旋转和翻转等形状变换
        取消随机掩码
    id 38 for yolov5 加上分类，保存txt
    id 39 将ok图由均衡化的图变为仿射图
v20:
    id 40 ok图换成长图，对多线程进行控制
'''


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
    radius = np.random.randint(min(w, h) // 6, min(w, h) // 5)

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

        axex_x = np.random.randint(w // 128, w // 8)
        axex_y = np.random.randint(h // 128, h // 8)

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
        self.trainlabelpath = os.path.join(save_dir, "trainlabel_id41.dict")
        self.testlabelpath = os.path.join(save_dir, "testlabel_id41.dict")

        # self.ngid = 0
        # self.save = False

    def analysis(self, roiimgdir):
        from scipy import stats

        sumlist = []
        hlist = []
        wlist = []
        for imgname in os.listdir(roiimgdir):
            imgpath = os.path.join(roiimgdir, imgname)
            img = cv2.imread(imgpath, 0)
            # print(img.shape)
            h, w = img.shape
            hlist.append(h)
            wlist.append(w)
            sumlist.append(h + w)
        print(max(hlist), min(hlist), max(wlist), min(wlist))
        plt.subplot(2, 2, 1)

        b = 100
        plt.hist(hlist, b)
        plt.subplot(2, 2, 2)

        plt.hist(wlist, b)
        plt.subplot(2, 2, 3)
        plt.hist(sumlist, b)

        m = np.mean(wlist)
        s = np.std(wlist, ddof=1)
        sumlist_ = (np.array(sumlist) - m) / s
        print(sumlist_)
        print(m, s)
        plt.subplot(2, 2, 4)
        plt.hist(sumlist_, b)
        # prob = stats.norm.pdf(sumlist_, m, s)
        # print(prob)
        plt.show()

    def precheck(self):
        for imgname in os.listdir(img_dir):
            imgpath = os.path.join(img_dir, imgname)
            img = cv2.imread(imgpath, 0)
            h, w = img.shape
            if min(h, w) < 2048:
                print(imgname)
                os.remove(imgpath)

    def select_ngimg(self, roiimgdir_):
        roiimgdir = os.path.join(roiimgbase_dir, roiimgdir_, "img2")
        roimaskdir = os.path.join(roiimgbase_dir, roiimgdir_, "mask2")

    def checkngimg(self):
        for subimgdir in os.listdir(roiimgbase_dir):
            print(subimgdir)
            roiimgdir = os.path.join(roiimgbase_dir, subimgdir, "img1")
            roimaskdir = os.path.join(roiimgbase_dir, subimgdir, "mask1")
            imgdirlist = os.listdir(roiimgdir)
            maskdirlist = os.listdir(roimaskdir)
            print(imgdirlist == maskdirlist)
            print(len(imgdirlist), len(maskdirlist))
            if imgdirlist != maskdirlist:
                for imgname in imgdirlist:
                    if imgname not in maskdirlist:
                        print(f"imgname: {imgname}")
                        rmpath = os.path.join(roiimgdir, imgname)
                        print(os.path.exists(rmpath))
                        # os.remove(rmpath)
                for imgname_ in maskdirlist:
                    if imgname_ not in imgdirlist:
                        print(f"imgname_, {imgname_}")
                        rmpath_ = os.path.join(roimaskdir, imgname_)
                        print(os.path.exists(rmpath_))
                        # os.remove(rmpath_)

    def get_ng(self, roiimgdir_):
        '''
         生成ng区域
        :param roiimgdir:
        :return:
        '''
        # assert roiimgdir != None
        # 选择生成ng的文件夹。默认 小黑缺陷
        self.flag2 = True if roiimgdir_.startswith("2") else False  # 亮白缺陷
        self.flag3 = True if roiimgdir_.startswith("3") else False  # 浅白缺陷
        self.flag1 = True if roiimgdir_.startswith("1") else False  # 大灰缺陷
        # self.flag4 = True if roiimgdir_.startswith("4") else False  # 鼓包缺陷
        cls = int(roiimgdir_[0])
        # 二类没有模糊图片
        # if self.flag2:
        #     isBlur = False

        # if isBlur:
        #     roiimgdir = os.path.join(roiimgbase_dir, roiimgdir_, "img2")
        #     roimaskdir = os.path.join(roiimgbase_dir, roiimgdir_, "mask2")
        # else:
        roiimgdir = os.path.join(roiimgbase_dir, roiimgdir_, "img1")
        roimaskdir = os.path.join(roiimgbase_dir, roiimgdir_, "img1/outputs/attachments")
        # exit()
        masknamelist = os.listdir(roimaskdir)
        len_imglist = len(masknamelist)

        # N = 1

        # 取ng图的索引
        i3 = np.random.randint(0, len_imglist)

        # 按高斯分布生成随机宽高
        mu = torch.ones(1) * 90
        sigma = torch.ones(1) * 20
        min_size = 60
        max_size = 120

        # if self.flag4:
        #     mu = torch.ones(1) * 1000
        #     sigma = torch.ones(1) * 1000
        #     min_size = 500
        #     max_size = 1500

        if self.flag2:
            mu = torch.ones(1) * 80
            sigma = torch.ones(1) * 20
            min_size = 60
            max_size = 100

        if self.flag1 or self.flag3:
            mu = torch.ones(1) * 200
            sigma = torch.ones(1) * 100
            min_size = 50
            max_size = 400
        # if self.flag1 or self.flag3:
        #     mu = torch.ones(1) * 150
        #     sigma = torch.ones(1) * 80
        #     min_size = 80
        #     max_size = 350

        while True:
            while True:
                w = torch.normal(mu, sigma).int().item()
                # h = torch.normal(mu, sigma).int().item()
                if w > min_size and w < max_size:
                    break

            while True:
                # w = torch.normal(mu, sigma).int().item()
                h = torch.normal(mu, sigma).int().item()
                if h > min_size and h < max_size:
                    break

            # if self.flagB:
            # 造的方一点，扁平化在下面部分处理。
            if ((h / w) < (3 / 2)) or ((h / w) > (2 / 3)):
                break

        # 按概率变换真正的形状， 五分之一为高，五分之一为平，其它为方
        # wh = np.random.randint(5)
        w_ = w
        h_ = h
        # if min(w, h) > 60:
        #     if wh % 5 == 0:
        #         w_ *= (np.random.rand() * 0.8 + 0.2)
        #         w_ = int(w_)
        #     elif wh % 5 == 1:
        #         h_ *= (np.random.rand() * 0.8 + 0.2)
        #         h_ = int(h_)
        # 选择ng图
        maskname = masknamelist[i3]
        imgname = maskname.replace("_1.png", ".jpg")
        maskpath = os.path.join(roimaskdir, maskname)
        imgpath = os.path.join(roiimgdir, imgname)
        # print(f"imgpath:{imgpath}, {os.path.exists(imgpath)}")
        # exit()
        # 打开图片
        img_ng = cv2.imread(imgpath, 0)
        img_mask = cv2.imread(maskpath, 0)
        # error 严重错误 resize 应为()
        # print(f"img:{img_ng.shape}")
        # exit()
        img_ng = cv2.resize(img_ng, (w_, h_))
        img_mask = cv2.resize(img_mask, (w_, h_))

        return img_ng, img_mask, imgname, cls

    def merge(self, srcimg, imgid, savedir, labelsavedir, labelpath):

        global INDEX
        # self.labelpath = labelpath
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
        # 选择基文件夹，不同分类下有img1代表不模糊， img2代表模糊
        roiimgdirs = os.listdir(roiimgbase_dir)
        i0 = []
        # 文件夹名的最后是概率
        for dirname in roiimgdirs:
            i0.append(float(dirname.split(".")[0].split("_")[-1]) / 100)
        #  按概率选择文件夹
        roiimgdir_N = torch.multinomial(torch.Tensor(i0), 1).item()
        roiimgdir = roiimgdirs[roiimgdir_N]
        # 设置选择模糊文件的概率40%
        # isblur = np.random.randint(10) > 5
        # isblur = 0
        # 保存的文件名
        savename = f"{roiimgdir[0]}_{imgid}_{self.index}.jpg"  # 原文件名已经实现了分类
        # print(f"savename: {savename}")
        labeldict = {}
        labeldict[savename] = []
        # 复制原ok文件
        dstimg = srcimg.copy()
        # 空白掩码
        # dstmask = np.zeros_like(dstimg, dtype=np.uint8)
        # 图片宽高
        height, width = srcimg.shape
        # h_ratio = IMGSIZE / height
        # w_ratio = IMGSIZE / width
        # max_ng = max(w, h)
        # 将原图片分成小格，确保ng区域不重复
        # 分割单元
        max_ng = 200  # 252 56 252 60 300->200
        x_len = width // max_ng
        y_len = height // max_ng
        idlist = []
        # coordlist = []
        # 一张图生成ng的最大数量
        ng_num_max = min(8, (x_len - 1) * (y_len - 1))
        # 一张图生成ng的实际数量， 随机生成，最小生成3个，一张图只生成一个类型的缺陷
        ng_num = np.random.randint(3, ng_num_max)

        for i in range(ng_num):
            # 计时，保证延时退出
            t1 = time.time()
            while True:
                # 所在格子的坐标
                x1_id_ = np.random.randint(1, x_len)
                y1_id_ = np.random.randint(1, y_len)
                # 可生成flag
                flag = False
                t2 = time.time()
                t = t2 - t1
                if t > 0.1:
                    # 超时退出
                    break
                # 不生成重复点
                for id in idlist:
                    xid, yid = id

                    # 不生成重复点
                    if np.abs(x1_id_ - xid) <= 1 and np.abs(y1_id_ - yid) <= 1:
                        flag = False
                        break
                    flag = True
                # 生成图片
                # print(f"flag:{flag}, {len(idlist)}")
                if flag or len(idlist) == 0:
                    # 记录生成的位置
                    idlist.append((x1_id_, y1_id_))
                    # 生成ng区域
                    ngimg, maskimg, imgname, cls = self.get_ng(roiimgdir)
                    # 去掉掩码特别小的
                    if (np.sum(maskimg > 10) < 100):
                        continue

                    h, w = ngimg.shape
                    # 坐标换算
                    x1 = x1_id_ * max_ng + np.random.randint(0, max(max_ng - w, 1))
                    y1 = y1_id_ * max_ng + np.random.randint(0, max(max_ng - h, 1))
                    x2 = x1 + w
                    y2 = y1 + h
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    # IMPORTANT 如果坐标不在图片内则不往下进行
                    if x2 >= width or y2 >= height:
                        continue
                    # # 字典记录数据 应该放在后面
                    # self.labeldict[savename].append((x1, y1, x2, y2))
                    # 得到roi区域
                    roiimg = srcimg[y1:y2, x1:x2]
                    # 排除全黑区域 BLAKTHRESH默认为10
                    black_mask = roiimg > BLACKTHRESH
                    if np.mean(black_mask) < 0.9:
                        continue

                    # 对原图和掩码作旋转和翻转等形状变换，各种参数随机生成
                    # ngimg, maskimg = img_converse2(ngimg, maskimg)
                    # 处理掩码
                    # ng掩码
                    ngimg = ngimg.astype(np.uint8)
                    mask_ng = (maskimg > MASKTHRESH)

                    # 膨胀
                    S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_SIZE, DILATE_SIZE))
                    mask_out = (cv2.dilate(mask_ng.astype(np.uint8), S) > 0)  # 之前版本严重错误之处
                    # 按掩码改权重

                    # complement mask
                    mask_out_compt = ~mask_out
                    # the median of ng image background
                    mask_out_compt_median = np.mean(ngimg[mask_out_compt])
                    if self.flag1:
                        # 原位置换方法
                        # the mean of roi img
                        roi_median = np.mean(roiimg)
                        # balance image
                        ngimg_ = ngimg * roi_median / mask_out_compt_median
                        # roi_weight = np.random.random(roiimg.shape) * np.random.uniform(0.05, 0.2)
                        roi_weight = np.random.uniform(0.05, 0.1)
                        roiimg[mask_out] = ngimg_[mask_out] * (1 - roi_weight) + roiimg[mask_out] * roi_weight
                    else:
                        # 模板方法
                        ngimg_normal = ngimg.astype(np.float32) / mask_out_compt_median
                        roiimg_ = roiimg.astype(np.float32) * ngimg_normal
                        roiimg_ = roiimg_.clip(0, 255).astype(np.uint8)
                        # 对粉尘斑作特殊处理
                        # if self.flag1:
                        #     roiimg_ = cv2.medianBlur(roiimg_, 11)
                        # 掩码作交集
                        # mask = mask_out

                        # 按mask数据复制到roi
                        roiimg[mask_out] = roiimg_[mask_out]

                    # 修改图片
                    dstimg[y1:y2, x1:x2] = roiimg
                    # dstmask[y1:y2, x1:x2][mask_out] = 255
                    # dstmask[mask] = 255
                    # 字典记录数据
                    # 框过大，缩小一些
                    w_ = w // 1
                    h_ = h // 1
                    # if w / h > 2:
                    #     w_ = w_ // 2
                    # if h / w > 2:
                    #     h_ = h_ // 2
                    # cx_ = int(cx * w_ratio)
                    # cy_ = int(cy * h_ratio)
                    # w_ = int(w_ * w_ratio)
                    # h_ = int(h_ * h_ratio)

                    cx_ = float(cx / width)
                    cy_ = float(cy / height)
                    w_ = float(w_ / width)
                    h_ = float(h_ / height)
                    labeldict[savename].append((f"{cls} {cx_} {cy_} {w_} {h_}"))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (10,)
                    thickness = 2
                    if ISTEST:
                        cv2.putText(dstimg, imgname, (x1, y1), font, 1, color, thickness)
                        cv2.rectangle(dstimg, (x1, y1), (x2, y2), (0), 1)
                    break

        if SAVE:
            if MULTITHREAD:
                LOCK.acquire()
            labeldict_ = torch.load(labelpath) if os.path.exists(labelpath) else {}
            labeldict_.update(labeldict)
            torch.save(labeldict_, labelpath)
            if MULTITHREAD:
                LOCK.release()
            savepath = os.path.join(savedir, savename)
            labelsavepath = os.path.join(labelsavedir, f"{savename.split('.')[0]}.txt")
            dstimg = dstimg.clip(0, 255).astype(np.uint8)
            # dstmask = dstmask.astype(np.uint8)
            # dstmask = np.clip(dstmask, 0, 255)
            dstimg = cv2.resize(dstimg, (WIDTHSIZE, HEIGHTSIZE))
            # dstmask = cv2.resize(dstmask, (MASKSIZE, MASKSIZE))
            cv2.imwrite(savepath, dstimg)
            with open(labelsavepath, 'w') as f:
                for line in labeldict[savename]:
                    print(line, file=f)
            # cv2.imwrite(masksavepath, dstmask)
        # self.index += 1

    # def generate_handle(self):

    def generate_train(self):

        print("gen train start")
        global INDEX
        imgdir = Path(img_dir)
        imglist = list(imgdir.glob("**/*.jpg"))
        imglistlen = len(imglist)

        trainsave_dir = os.path.join(self.savedir, "images")
        labelsave_dir = os.path.join(self.savedir, "labels")

        self.labeldict = {}
        if NEWGEN:
            removepath(trainsave_dir)
            removepath(labelsave_dir)
            if os.path.exists(self.trainlabelpath):
                os.remove(self.trainlabelpath)
        else:
            if os.path.exists(self.trainlabelpath):
                self.labeldict = torch.load(self.trainlabelpath)

        makedir(trainsave_dir)
        makedir(labelsave_dir)

        INDEX = min(len(os.listdir(trainsave_dir)), len(os.listdir(labelsave_dir)))

        tids = []
        while (INDEX < TRAINNUM):
            idxs = np.random.randint(0, imglistlen)
            imgpath = imglist[idxs]
            imgid = f"{imgpath.parent.stem}_{imgpath.stem}"
            # print(imgid)
            # exit()
            # print(imgpath)
            # print(imgid)
            # print(imgpath.parent.name)
            # exit(1026)
            srcimg = cv2.imread(str(imgpath), 0)
            # for i in range(THREADNUM):
            # self.merge(srcimg, imgid, trainsave_dir, labelsave_dir)
            # tid = threading.Thread(target=self.merge,
            #                        args=[srcimg, imgid, trainsave_dir, labelsave_dir, self.trainlabelpath])
            # # print(f"tid: {tid.name}")
            # tids.append(tid)
            # tid.start()
            # if len(tids) >= THREADNUM:
            #     tids = []
            #     for tid in tids:
            #         tid.join()
            # 这么写容易造成两个函数的线程交叉污染
        #     if len(tids) >= THREADNUM:
        #         for tid in tids:
        #             tid.join()
        #         tids = []  # important
        # for tid in tids:
        #     tid.join()
        # tids = []
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



    def generate_test(self):

        print("gen test start")
        global INDEX
        imgdir = Path(img_dir)
        imglist = list(imgdir.glob("**/*.jpg"))
        imglistlen = len(imglist)
        testsave_dir = os.path.join(self.savedir, "testimages")
        testlabelsave_dir = os.path.join(self.savedir, "testlabels")
        self.labeldict = {}
        if NEWGEN:
            removepath(testsave_dir)
            removepath(testlabelsave_dir)
            if os.path.exists(self.testlabelpath):
                os.remove(self.testlabelpath)
        else:
            if os.path.exists(self.testlabelpath):
                self.labeldict = torch.load(self.testlabelpath)

        makedir(testsave_dir)
        makedir(testlabelsave_dir)
        INDEX = min(len(os.listdir(testsave_dir)), len(os.listdir(testsave_dir)))
        tids = []
        while (INDEX < TESTNUM):
            idxs = np.random.randint(0, imglistlen)
            imgpath = imglist[idxs]
            imgid = f"{imgpath.parent.stem}_{imgpath.stem}"

            srcimg = cv2.imread(str(imgpath), 0)
            # for i in range(THREADNUM):
            # self.merge(srcimg, imgid, trainsave_dir, labelsave_dir)
            # tid = threading.Thread(target=self.merge,
            #                        args=[srcimg, imgid, trainsave_dir, labelsave_dir, self.trainlabelpath])
            # # print(f"tid: {tid.name}")
            # tids.append(tid)
            # tid.start()
            # if len(tids) >= THREADNUM:
            #     tids = []
            #     for tid in tids:
            #         tid.join()
            # 这么写容易造成两个函数的线程交叉污染
        #     if len(tids) >= THREADNUM:
        #         for tid in tids:
        #             tid.join()
        #         tids = []  # important
        # for tid in tids:
        #     tid.join()
        # tids = []
            if MULTITHREAD:
                tid = threading.Thread(target=self.merge,
                                       args=[srcimg, imgid, testsave_dir, testlabelsave_dir, self.testlabelpath])
                tids.append(tid)
                tid.start()
                if len(tids) >= THREADNUM:
                    tids = []
                    for tid in tids:
                        tid.join()
            else:
                self.merge(srcimg, imgid, testsave_dir, testlabelsave_dir, self.testlabelpath)
        # while (INDEX < TESTNUM):
        #     index = np.random.randint(0, imgdir_listlen)
        #     imgname = imgdir_list[index]
        #     imgid = imgname.split(".")[0]
        #     imgpath = os.path.join(imgdir, imgname)
        #     srcimg = cv2.imread(imgpath, 0)
        #     # self.merge(srcimg, imgid, testsave_dir, testlabelsave_dir)
        #     tid = threading.Thread(target=self.merge,
        #                            args=[srcimg, imgid, testsave_dir, testlabelsave_dir, self.testlabelpath])
        #     tids.append(tid)
        #     tid.start()
        #     if len(tids) >= THREADNUM:
        #         for tid in tids:
        #             tid.join()
        #         tids = []  # important
        # for tid in tids:
        #     tid.join()
        # tids = []

    def generate(self):
        self.generate_train()
        time.sleep(1)  # 在一定程度上抑制两个函数间线程的交叉污染问题
        # self.generate_test()

    def load_label(self):
        traindict = torch.load(self.trainlabelpath)
        # testdict = torch.load(self.testlabelpath)
        print(f"dictlen: {len(traindict)}")
        print(traindict)
        # exit(1107)
        # print(traindict.keys())
        # print(testdict.keys())
        # print(traindict)
        # print(testdict)
        train_dir = os.path.join(self.savedir, "images")
        save_dir = os.path.join(self.savedir, "dictsave")
        removepath(save_dir)
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
            savepath = os.path.join(save_dir, imgname)

            # print(savepath)
            for coord in traindict[imgname]:
                cls, cx, cy, w, h = [float(x) for x in coord.split(" ")]
                cx = int(cx*WIDTHSIZE)
                cy = int(cy*HEIGHTSIZE)
                w = int(w*WIDTHSIZE)
                h = int(h*HEIGHTSIZE)
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
    #     print("sys.argv", len(sys.argv))
    #     if (len(sys.argv)>1):
    #         self.generate_test()
    #     else:
    #         self.generate_train()


if __name__ == '__main__':
    an = AugNeg()
    t1 = time.time()
    an.generate()
    # an.load_label()
    t2 = time.time()
    print(f"time: {t2 - t1}")
    # an.checkngimg()

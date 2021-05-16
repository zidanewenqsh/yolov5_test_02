# #!/e/Soft/Python/Python38/python
# # -*- coding: utf-8 -*- #
# # ------------------------------------------------------------------
# # @File Name:        get_yolov5_anchor_class.py
# # @Author:           wen
# # @Version:          ver0_1
# # @Created:          2021/3/2 10:44
# # @Description:      注意尺寸缩放的问题
# # note: 训练集的大小可以改变
# # Function List:     hello() -- print helloworld
# # History:
# #       <author>    <version>   <time>      <desc>
# #       wen         ver0_1      2020/12/15  xxx
# # ------------------------------------------------------------------
# import os
# import numpy as np
# import torch
# import cv2
# import time
# from pathlib import Path
# from PIL import Image
# from sklearn.cluster import KMeans
# from utils.general import xywh2xyxy
# import yaml
# import logging
# from tqdm import tqdm
#
# class Anchor_Generator:
#     def __init__(self, basedir, whsavepath, yamlpath=None, maxnum=5000, iouthresh = 0.2, width=640, height=640):
#         logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
#         self.basedir = Path(basedir)
#         self.yamlpath = Path(yamlpath) if yamlpath else None
#         self.iouthresh = iouthresh
#         self.images_dir = self.basedir / "images"
#         self.labels_dir = self.basedir / "labels"
#         # if whsavepath:
#         self.whsavepath = Path(whsavepath)
#         if self.whsavepath.exists():
#             self.labelarr = np.load(self.whsavepath)
#             logging.info("load sucessfully")
#         else:
#             imgfilelist = list(self.images_dir.glob("*.jpg"))
#             labelfilelist = list(self.labels_dir.glob("*.txt"))
#             maxnum = min(maxnum, len(labelfilelist))
#             ids = torch.randperm(len(imgfilelist))[:maxnum].tolist()
#             labelfilelist = np.array(labelfilelist)[ids]
#             labeldict = {x.stem: self.getlabelinfo(x, width, height) for x in labelfilelist if len(self.getlabelinfo(x, width, height)) > 0}
#             self.labelarr = np.concatenate(list(labeldict.values()))
#             np.save(self.whsavepath, self.labelarr)
#             logging.info("save sucessfully")
#
#     @staticmethod
#     def mybox_iou(box1, box2):
#         '''
#
#         Args:
#             box1:
#             box2:
#
#         Returns:
#
#         '''
#         box1 = box1[..., None, :]
#         inter = np.minimum(box1, box2).prod(-1)
#         union = np.maximum(box1, box2).prod(-1)
#         return inter / union
#
#     @staticmethod
#     def getlabelinfo(labelpath, width, height):
#         '''
#
#         Args:
#             labelpath:
#
#         Returns:
#
#         '''
#         # imgpath = Path(str(labelpath).replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep, 1)).with_suffix(
#         #     ".jpg")
#         # with Image.open(imgpath) as img:
#         #     width, height = img.size
#         with open(labelpath) as f:
#             labelarr = np.array([x.split(" ")[3:] for x in f.read().split("\n") if x != '']).astype(np.float32)
#         if len(labelarr) > 0:
#             labelarr *= np.array([width, height])
#         return labelarr.astype(np.int64)  # 返回带宽高的array
#
#     def labelarr_nms(self, labelarr, num):
#         '''
#         通过聚类的方法，将数据分类，取出平均值iou最大的返回，并将其从原数据中去除
#         Args:
#             labelarr:
#             num:
#
#         Returns:
#
#         '''
#         if len(labelarr) < num * 2:
#             return np.array([]), np.array([])
#
#         y_pred = KMeans(n_clusters=num).fit_predict(labelarr)
#
#         ylist = []
#
#         for i in range(num):
#             mask = y_pred == i
#             y = labelarr[mask]
#             ylist.append(np.mean(y, axis=0))
#         yarr = np.stack(ylist).astype(np.int64)
#         biou = self.mybox_iou(yarr, labelarr)
#         mask = (biou < iouthresh)
#         ids = mask.sum(1).argmin()
#         mask_ = mask[ids]
#
#         return labelarr[mask_], yarr[ids]
#
#     def get_boxiou_accuracy(self, box1, box2):
#         '''
#         计算所选的建议框和图像标签的iou最大值大于阈值的精度，反映了建议框选择的好坏
#         Args:
#             box1: 建议框
#             box2: 图像标签
#
#         Returns:
#
#         '''
#         iou = self.mybox_iou(box1, box2)
#         return np.mean(np.max(iou, axis=-1) > self.iouthresh)
#
#     def get_anchor(self):
#         '''
#         主方法
#         Returns:
#
#         '''
#         labelarr = self.labelarr.copy()
#         yarrlist = []
#         for _ in tqdm(range(15)):
#             labelarr, yarr = self.labelarr_nms(labelarr, num=12)
#             if len(labelarr) == 0:
#                 break
#             else:
#                 yarrlist.append(yarr)
#         anchor = np.stack(yarrlist)
#         ids = anchor.mean(1).argsort()
#         anchor = anchor[ids]
#         accu = self.get_boxiou_accuracy(anchor, self.labelarr)
#         logging.info(f"accuracy:{accu}")
#         if self.yamlpath and len(anchor)%3==0:
#             with open(yamlpath, 'w') as f:
#                 yaml.dump({"anchors":anchor.reshape(len(anchor)//3, -1).tolist()}, f)
#             logging.info("yaml dump successfully")
#         return anchor
#
#     def __call__(self, *args, **kwargs):
#         return self.forward()
#
#     def forward(self):
#         return self.get_anchor()
#     # @staticmethod
#     # def print_anchor(anchor_: np.ndarray):
#     #     ids = anchor_.mean(1).argsort()
#     #     anchor = anchor_[ids]
#     #     print(anchor)
#
#
# if __name__ == '__main__':
#     mode = 1
#     if mode:
#         iouthresh = 0.33 # for woods
#         # base路径
#         basedir = Path(r"F:\FProjectsData\woods05\datas\Auguments\saves\Aug_woods_ng_id41\train")
#         # 保存的路径
#         name = "mywoodslabel00"
#         height = 640
#         width = 640//4
#     else:
#         iouthresh = 0.36 # for dents
#         basedir = Path(r"F:\FProjectsData\dents06\datas\Augments\saves\dents_yolov5_id21\train")
#         name = "mydentslabel00"
#         height = 640
#         width = 374
#     whsavepath = Path(f"{name}.npy")
#     yamlpath = Path(f"{name}.yaml")
#     if whsavepath.exists():
#         whsavepath.unlink()
#     ag = Anchor_Generator(basedir, whsavepath, yamlpath, maxnum=1000,iouthresh=iouthresh, width=width,height=height)
#     anchor = ag()
#     print(anchor)
#     # p1 = r"F:\AI-046\Projects\yolov5_test_01\models\yolov5s.yaml"
#     # p1 = r"F:\AI-046\Projects\yolov5_test_01\myutils\mydentslabel01.yaml"
#     # with open(p1) as f:
#     #     h1 = yaml.load(f, Loader=yaml.FullLoader)
#     # print(h1)
#     # exit()
#     # print(h1['anchors'])
#     # a = h1['anchors']
#     # print(type(a))
#     # with open("temp.yaml", 'a') as f:
#     #     yaml.dump(a, f)
#
#     # 'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
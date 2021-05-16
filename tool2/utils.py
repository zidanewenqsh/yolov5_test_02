import time
from PIL import Image, ImageDraw
import torch.nn as nn
import torch
import numpy as np
import os
import socket
import struct

import sys
from PIL import Image
from torchvision import transforms
import shutil

g_transform = transforms.Compose([
    transforms.ToTensor()
])
# print("****")
# print(__file__)
# print(os.path.dirname(__file__))
# print("****")
'''
version 3'''

def encode(ip_addr=None, ip_port=8080, start=None, **kwargs):
    # print(ip_addr)
    if ip_addr == None:
        # 获取计算机名称
        hostname = socket.gethostname()
        # 获取本机IP
        ip = socket.gethostbyname(hostname)
        ip_addr = ip
    # else:
    #     ip_addr = "127.0.0.1"
    if ip_port==None:
        ip_port = 8080

    SEND_BUF_SIZE = 4096

    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(ip_addr, ip_port)
    p.connect((ip_addr, ip_port))

    epathlist = []
    inextlist = []
    exextlist = []
    excludedir = []

    for k, v in kwargs.items():
        assert isinstance(v, list)
        if k == "epath":
            epathlist.extend(v)
        if k == "inext":
            inextlist.extend(v)
        if k == "exext":
            exextlist.extend(v)
        if k == "exdir":
            excludedir.extend(v)

    for epath in epathlist:
        epath = epath.replace('\\', '/')
        epath_len = len(epath.split('/'))
        start = epath_len -1 if start==None else start
        for roots, dirs, files in os.walk(epath):
            roots = roots.replace('\\', '/')
            rootslist = roots.split('/')
            rootslist_len = len(rootslist)
            rel_len = rootslist_len - epath_len
            # print("rel_len", rel_len)
            dir_1 = rootslist[-1] if rel_len >0 else None
            dir_2 = rootslist[-2] if rel_len >1 else None
            dir_3 = rootslist[-3] if rel_len >2 else None
            # print(dir_1, dir_2, dir_3)
            if dir_1 in excludedir or dir_2 in excludedir or dir_3 in excludedir:
                continue
            for file in files:
                ext = file.split('.')[-1]
                if ext in exextlist:
                    continue
                if ext in inextlist or len(inextlist) == 0:

                    filepath = os.path.join(roots, file).replace("\\", '/')

                    pathlist = filepath.split('/')
                    rel_path = '/'.join(pathlist[start:]).encode('utf-8')
                    relpath_len = len(rel_path)
                    with open(filepath, 'rb') as f:
                        buffer = f.read()
                        buffer_size = len(buffer)
                        mtime = int(os.stat(filepath).st_mtime)
                        fmt = f">H3I"
                        msg = struct.pack(fmt, 12345, mtime, relpath_len, buffer_size)

                        p.sendall(msg)
                        msg = struct.pack(f">{relpath_len}s", rel_path)
                        p.sendall(msg)

                        flagsize = struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]

                        if flag:
                            msg = struct.pack(f">{buffer_size}s", buffer)

                            if buffer_size > SEND_BUF_SIZE:
                                send_index = 0

                                while send_index < buffer_size:
                                    send_size = min(SEND_BUF_SIZE, buffer_size - send_index)
                                    p.sendall(msg[send_index:send_index + send_size])
                                    send_index += send_size
                            else:
                                p.sendall(msg)
                            print(f"send file: {filepath}")
                        else:
                            print(f"pass file: {filepath}")

    p.close()


def encode_v3(addr="192.168.1.195", port=8080, start=4, **kwargs):
    SEND_BUF_SIZE = 4096

    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ip_addr = '192.168.1.195'
    ip_addr = addr
    ip_port = port
    p.connect((ip_addr, ip_port))  # 要先连

    epathlist = []

    inextlist = []
    exextlist = ['tors', 'pt', 'pth', 'tar', 'tar.gz', 'script', "pyc", "scriptx"]
    excludedir = ["datas", "saves", "save", '.idea', '.git']
    for k, v in kwargs.items():
        assert isinstance(v, list)
        if k == "epath":
            epathlist.extend(v)
        if k == "inext":
            inextlist.extend(v)
        if k == "exext":
            exextlist.extend(v)
        if k == "exdir":
            excludedir.extend(v)

    for epath in epathlist:
        for roots, dirs, files in os.walk(epath):

            dir_1 = roots.split('\\')[-1]
            dir_2 = roots.split('\\')[-2]
            dir_3 = roots.split('\\')[-3]
            # print(dir_1)
            if dir_1 in excludedir or dir_2 in excludedir or dir_3 in excludedir:
                continue
            for file in files:
                ext = file.split('.')[-1]
                if ext in exextlist:
                    continue
                if ext in inextlist or len(inextlist) == 0:

                    filepath = os.path.join(roots, file).replace("\\", '/')
                    print(filepath)
                    pathlist = filepath.split('/')
                    rel_path = '/'.join(pathlist[start:]).encode('utf-8')
                    relpath_len = len(rel_path)
                    with open(filepath, 'rb') as f:
                        buffer = f.read()  # 要与writelines配对
                        buffer_size = len(buffer)
                        mtime = int(os.stat(filepath).st_mtime)
                        fmt = f">H3I"
                        msg = struct.pack(fmt, 12345, mtime, relpath_len, buffer_size)

                        p.sendall(msg)
                        msg = struct.pack(f">{relpath_len}s", rel_path)
                        p.sendall(msg)

                        flagsize =struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]

                        if flag:
                            msg = struct.pack(f">{buffer_size}s", buffer)
                            # print("msg1", len(msg))
                            if buffer_size > SEND_BUF_SIZE:
                                send_index = 0

                                while send_index < buffer_size:
                                    send_size = min(SEND_BUF_SIZE, buffer_size - send_index)
                                    p.sendall(msg[send_index:send_index + send_size])
                                    send_index += send_size
                            else:
                                p.sendall(msg)
                            # print("send success")
                            print(f"send file: {filepath}")
    p.close()
    # break
    # if msg == '1':
    #     break
    # f.close()


def encode_v2(addr, start, **kwargs):
    SEND_BUF_SIZE = 4096

    dictpath = "D:/AI-046/temp/temp.tor"
    datadict = {}
    epathlist = []
    # start = 0
    # startlist = []
    inextlist = []
    exextlist = ['tors', 'pt', 'pth', 'tar', 'tar.gz', 'script', "pyc", "scriptx"]
    excludedir = ["datas", "saves", "save", '.idea', '.git']
    for k, v in kwargs.items():
        assert isinstance(v, list)
        if k == "epath":
            epathlist.extend(v)
        if k == "inext":
            inextlist.extend(v)
        if k == "exext":
            exextlist.extend(v)
        if k == "exdir":
            excludedir.extend(v)

    for epath in epathlist:
        for roots, dirs, files in os.walk(epath):
            # print(roots)
            # exit()
            dir_1 = roots.split('\\')[-1]
            dir_2 = roots.split('\\')[-2]
            dir_3 = roots.split('\\')[-3]
            # print(dir_1)
            if dir_1 in excludedir or dir_2 in excludedir or dir_3 in excludedir:
                continue
            for file in files:
                ext = file.split('.')[-1]
                if ext in exextlist:
                    continue
                if ext in inextlist or len(inextlist) == 0:
                    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # ip_addr = '192.168.1.195'
                    ip_addr = addr
                    ip_port = 8080
                    p.connect((ip_addr, ip_port))  # 要先连
                    filepath = os.path.join(roots, file).replace("\\", '/')
                    print(filepath)
                    pathlist = filepath.split('/')
                    # print(pathlist[2:])
                    # print("pathllist",pathlist)
                    rel_path = '/'.join(pathlist[start:]).encode('utf-8')
                    relpath_len = len(rel_path)
                    with open(filepath, 'rb') as f:
                        buffer = f.read()  # 要与writelines配对
                        buffer_size = len(buffer)
                        # datadict[key] = torchdata
                        # break
                        # torch.save(datadict, dictpath)

                        # flag = 1
                        mtime = int(os.stat(filepath).st_mtime)
                        fmt = f">H3I"
                        msg = struct.pack(fmt, 12345, mtime, relpath_len, buffer_size)

                        p.sendall(msg)
                        msg = struct.pack(f">{relpath_len}s", rel_path)
                        p.sendall(msg)

                        flagsize =struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]
                        # print("flag", flag)
                        if flag:
                            # print("continue")
                            continue

                        msg = struct.pack(f">{buffer_size}s", buffer)
                        # print("msg1", len(msg))
                        if buffer_size > SEND_BUF_SIZE:
                            send_index = 0

                            while send_index < buffer_size:
                                send_size = min(SEND_BUF_SIZE, buffer_size - send_index)
                                p.sendall(msg[send_index:send_index + send_size])
                                send_index += send_size
                        else:
                            p.sendall(msg)
                        # p.send(msg)
                        # flag = 1
                        # fmt = f">H2I{buffer_size}s"
                        # msg = struct.pack(fmt, 12345, flag, buffer_size, buffer)
                        # msg = struct.pack(f">H{buffer_size}sI", 12345, buffer, flag)
                        # msg = struct.pack(">I", start)

                        # p.send(msg)
                        # p.send(msg.encode('utf-8'))  # 收发消息一定要二进制，记得编码
                        print("send success")
                        time.sleep(0.1)
                        p.close()
    # break
    # if msg == '1':
    #     break
    # f.close()

def encode_wrong(start, **kwargs):
    SEND_BUF_SIZE = 4096
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip_addr = '192.168.1.195'
    ip_port = 8080
    p.connect((ip_addr, ip_port))  # 要先连
    epathlist = []
    # start = 0
    # startlist = []
    inextlist = []
    exextlist = ['tors', 'pt', 'pth', 'tar', 'tar.gz', 'script', "pyc", "scriptx"]
    excludedir = ["datas", "saves", "save", '.idea', '.git']
    for k, v in kwargs.items():
        assert isinstance(v, list)
        if k == "epath":
            epathlist.extend(v)
        if k == "inext":
            inextlist.extend(v)
        if k == "exext":
            exextlist.extend(v)
        if k == "exdir":
            excludedir.extend(v)

    for epath in epathlist:
        for roots, dirs, files in os.walk(epath):
            # print(roots)
            # exit()
            dir_1 = roots.split('\\')[-1]
            dir_2 = roots.split('\\')[-2]
            dir_3 = roots.split('\\')[-3]
            # print(dir_1)
            if dir_1 in excludedir or dir_2 in excludedir or dir_3 in excludedir:
                continue
            for file in files:
                ext = file.split('.')[-1]
                if ext in exextlist:
                    continue
                if ext in inextlist or len(inextlist) == 0:
                    # p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # ip_addr = '192.168.1.195'
                    # ip_port = 8080
                    # p.connect((ip_addr, ip_port))  # 要先连
                    filepath = os.path.join(roots, file).replace("\\", '/')
                    print(filepath)
                    pathlist = filepath.split('/')
                    # print(pathlist[2:])
                    # print("pathllist",pathlist)
                    rel_path = '/'.join(pathlist[start:]).encode('utf-8')
                    relpath_len = len(rel_path)
                    with open(filepath, 'rb') as f:
                        buffer = f.read()  # 要与writelines配对
                        buffer_size = len(buffer)
                        flag = 1
                        fmt = f">H3I"
                        msg = struct.pack(fmt, 12345, flag, relpath_len, buffer_size)

                        p.sendall(msg)
                        msg = struct.pack(f">{relpath_len}s", rel_path)
                        p.sendall(msg)

                        msg = struct.pack(f">{buffer_size}s", buffer)
                        if buffer_size > SEND_BUF_SIZE:
                            send_index = 0

                            while send_index < buffer_size:
                                send_size = min(SEND_BUF_SIZE, buffer_size - send_index)
                                p.sendall(msg[send_index:send_index + send_size])
                                send_index += send_size
                        else:
                            p.sendall(msg)
                        # p.send(msg)
                        # flag = 1
                        # fmt = f">H2I{buffer_size}s"
                        # msg = struct.pack(fmt, 12345, flag, buffer_size, buffer)
                        # msg = struct.pack(f">H{buffer_size}sI", 12345, buffer, flag)
                        # msg = struct.pack(">I", start)

                        p.send(msg)
                        # p.send("\0".encode('utf-8'))
                        # p.send(msg.encode('utf-8'))  # 收发消息一定要二进制，记得编码
                        print("send success")
                        # time.sleep(0.1)
    # flag = 0
    fmt = f">H3I"
    msg = struct.pack(fmt, 12345, 0, 0, 0)
    p.close()
    # break
    # if msg == '1':
    #     break
    # f.close()





def rmmkdir(dirpath):
    # if os.path.isdir(dirpath):
    removepath(dirpath)
    time.sleep(0.1)
    makedir(dirpath)


def makedir(path):
    '''

    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def removepath(path):
    '''

    :param path:
    :return:
    '''
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def img2Tensor(imgpath) -> torch.Tensor:
    '''

    :param imgpath:
    :return:
    '''
    with Image.open(imgpath) as img:
        img = img.convert("L")
        return g_transform(img)


def formattime(start_ms, end_ms):
    ms = end_ms - start_ms
    m_end, s_end = divmod(ms, 60)
    h_end, m_end = divmod(m_end, 60)
    time_data = "%02d:%02d:%02d" % (h_end, m_end, s_end)
    return time_data


def toTensor(data):
    '''
    :param data:
    :return:
    '''
    if isinstance(data, torch.FloatTensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, (list, tuple)):
        return torch.tensor(list(data)).float()  # 针对列表和元组，注意避免list里是tensor的情况
    elif isinstance(data, torch.Tensor):
        return data.float()
    return


def toNumpy(data):
    '''
    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(list(data))  # 针对列表和元组
    return


def toList(data):
    '''
    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return list(data)  # 针对列表和元组
    return


def isBox(box):
    '''
    判断是否是box
    :param box:
    :return:
    '''
    box = toNumpy(box)
    if box.ndim == 1 and box.shape == (4,) and np.less(box[0], box[2]) and np.less(box[1], box[3]):
        return True
    return False


def isBoxes(boxes):
    '''
    判断是否是boxes
    :param boxes:
    :return:
    '''
    # print(boxes)
    boxes = toNumpy(boxes)
    # print(boxes)
    # exit()
    if boxes.ndim == 2 and boxes.shape[1] == 4:
        # print("boxes.ndim == 2 and boxes.shape[1] == 4", boxes[np.greater_equal(boxes[:, 0], boxes[:, 2])])
        if np.less(boxes[:, 0], boxes[:, 2]).all() and np.less(boxes[:, 1], boxes[:, 3]).all():
            return True
    return False


def area(box):
    '''

    :param box:
    :return:
    '''
    return torch.mul((box[2] - box[0]), (box[3] - box[1]))


def areas(boxes):
    '''

    :param boxes:
    :return:
    '''
    return torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))


# iou


def iou(box, boxes, isMin=False):
    '''
    define iou function
    :param box:
    :param boxes:
    :param isMin:
    :return:
    '''

    box = toTensor(box)

    boxes = toTensor(boxes)  # 注意boxes为二维数组

    # 如果boxes为一维，升维
    if boxes.ndimension() == 1:
        boxes = torch.unsqueeze(boxes, dim=0)

    # box_area = torch.mul((box[2] - box[0]), (box[3] - box[1]))  # the area of the first row
    # boxes_area = torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))  # the area of other row

    box_area = area(box)
    boxes_area = areas(boxes)
    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.min(box[2], boxes[:, 2])
    yy2 = torch.min(box[3], boxes[:, 3])

    inter = torch.mul(torch.max(
        (xx2 - xx1), torch.Tensor([0, ])), torch.max((yy2 - yy1), torch.Tensor([0, ])))
    # print("inter",inter.shape, box_area.shape, boxes_area.shape, box_area)

    if (isMin == True):
        # intersection divided by union
        over = torch.div(inter, torch.min(box_area, boxes_area))
    else:
        # intersection divided by union
        over = torch.div(inter, (box_area + boxes_area - inter))
    return over


def nms(boxes_input, threshold=0.3, isMin=False):
    '''
    define nms function
    :param boxes_input:
    :param isMin:
    :param threshold:
    :return:
    '''
    # print("aaa",boxes_input[:,:4].shape)
    if isBoxes(boxes_input[:, :4]):
        '''split Tensor'''
        boxes = toTensor(boxes_input)

        boxes = boxes[torch.argsort(-boxes[:, 4])]
        r_box = []
        while (boxes.size(0) > 1):
            r_box.append(boxes[0])
            mask = torch.lt(iou(boxes[0], boxes[1:], isMin), threshold)
            boxes = boxes[1:][mask]  # the other row of Tensor
            '''mask 不能直接放进来,会报IndexError'''
        if (boxes.size(0) > 0):
            r_box.append(boxes[0])
        if r_box:
            return torch.stack(r_box)  # 绝对不能转整数，要不然置信度就变成0
    elif isBox(boxes_input):
        return toTensor(boxes_input)
    return torch.Tensor([])
    # return torch.stack(r_box).long()

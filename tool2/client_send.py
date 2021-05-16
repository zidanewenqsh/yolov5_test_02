#！/usr/bin/python
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

epathlist = [
    "D:\AI-046\Projects\woods03\saves",
    # "D:\AI-046\Projects\PHZN_api_01",
]
inextlist = ['py','c','cpp','h','txt', 'ini', 'md']
exextlist = ['tors', 'pt', 'pth', 'tar', 'tar.gz', 'script', "pyc", "scriptx"]
excludedir = ["datas",  '.idea', '.git']
if __name__ == '__main__':
    if len(exextlist)==0:
        exit(0)
    argc=len(sys.argv)
    # ip_addr = "127.0.1.1"
    ip_addr = "192.168.1.195"
    ip_port = 8080
    start = 2
    if argc>1:
        ip_addr = sys.argv[1]
    if argc>2:
        ip_port = int(sys.argv[2])
    if argc>3:
        start = int(sys.argv[3])
    encode(ip_addr, ip_port, start, epath=epathlist, inext=inextlist, exext=exextlist, exdir=excludedir)
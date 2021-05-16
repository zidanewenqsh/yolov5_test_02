#!/bin/python3
import os
import torch
def makedir(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
SAVEDIR = r"/home/wen/Projects/temp"
makedir(SAVEDIR)
dictpath = "/mnt/hgfs/AI-046/temp/temp.tor"
datadict = torch.load(dictpath)
for subpath, filedata in datadict.items():
    path = os.path.join(SAVEDIR, subpath).replace("\\", "/")
    filedir = os.path.dirname(path)
    makedir(filedir)
    print(path)
    with open(path, 'wb') as f:
        f.writelines(filedata)

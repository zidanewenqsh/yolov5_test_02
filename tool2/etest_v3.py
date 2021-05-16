import os
import torch
import sys
####################
# 从外部接受参数，路径和start
#####################
# from tool2.utils import makedir

# SAVEDIR = r"D:/AI-046/Projects/dents01/saves/saves0"
dictpath = "D:/AI-046/temp/temp.tor"
# print(os.path.exists(dictpath))
# exit()
# FILEDIR = os.getcwd()
# CODESDIR = os.path.dirname(FILEDIR)
# BASEDIR = os.path.dirname(CODESDIR)
datadict = {}

# epath = CODESDIR

# epath7 = r"D:\AI-046\PycharmProject\yolov3_01_"
argc = len(sys.argv)
epathlist = []
start = 0
# for i in range(1, argc):
#     epathlist.append(sys.argv[i])
if argc>2:
    epathlist.append(sys.argv[1])
    start = int(sys.argv[2])


# endlist = ['cpp','h','txt','py','jpg','png','c']
endlist = []
excludelist = ['tors', 'pt', 'pth', 'tar', 'tar.gz', 'script', "pyc", "scriptx"]
excludedir = ["datas", "saves", "save", '.idea', '.git']
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
            end = file.split('.')[-1]
            if end in excludelist:
                continue
            if end in endlist or len(endlist) == 0:
                filepath = os.path.join(roots, file).replace("\\", '/')
                print(filepath)
                pathlist = filepath.split('/')
                # print(pathlist[2:])
                # print("pathllist",pathlist)
                key = '/'.join(pathlist[start:])
                with open(filepath, 'rb') as f:
                    torchdata = f.readlines()
                    datadict[key] = torchdata
                # break
torch.save(datadict, dictpath)

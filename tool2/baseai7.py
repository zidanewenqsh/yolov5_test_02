import configparser
import matplotlib.pyplot as plt
import argparse
import os
import abc
import sys
import torch
import torch.nn as nn
import time
import numpy as np
import traceback
from torch import optim
from torch.utils import data
from torchvision import transforms

# import shutil
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FILEPATH = os.path.dirname(__file__)
CODESPATH = os.path.dirname(FILEPATH)
BASEPATH = os.path.dirname(CODESPATH)

if CODESPATH not in sys.path:
    sys.path.append(CODESPATH)

from tool2.utils import makedir, removepath

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseNet, self).__init__()
        self.name = self.__class__.__name__
        # print(__file__)
        self.basefile = __file__

    def paraminit(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)

    def forward(self, *args, **kwargs):
        pass


class BaseDataset(data.Dataset):
    def __init__(self):
        self.dataset = []

    def __str__(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pass


class BaseAI(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def _argparserinit(self):
        pass

    def _cfginit(self, cfgfile):
        pass

    def _loginit(self, test=True):
        pass

    def _log(self, isTest: bool = False, **kwargs):
        pass

    def _get_time(self):
        return time.strftime("%b/%d/%Y %H:%M:%S", time.localtime())

    def _get_var(self, paramLogFile=None):
        varc = vars(self).copy()

        poplist = []
        typelist = [int, float, dict, str, bool]
        for varkey in varc.keys():
            if "__" in varkey or 'log' in varkey or type(varc[varkey]) not in typelist:
                poplist.append(varkey)

        for popitem in poplist:
            varc.pop(popitem)

        paramdict = {}
        diffdict = {}
        if os.path.exists(paramLogFile):
            paramdict = torch.load(paramLogFile)

        for varkey in varc.keys():
            if varkey not in paramdict.keys() or paramdict[varkey] != varc[varkey]:
                diffdict[varkey] = varc[varkey]

        return varc, diffdict

    def tryfunc(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            # exc = traceback.format_exc()
            # self.printlog(exc)
            # traceback.print_exc()
            # traceback.format_exc()
            # self.printlog(f"error:, {os.path.basename(__file__)}, {e}, {excinfo[2].tb_lineno}")
            # print(sys.exc_info())
            exit(-1)

    def loginfo(self, **kwargs):
        pass

    def printlog(self, *args, newline=True):
        pass

    def _deviceinit(self):
        pass

    def _moduleinit(self):
        pass

    def _title(self):
        pass

    def otherinit(self):
        pass


class BaseTrain(BaseAI):
    '''
    baseai7:
    the seventhth version of BaseTrain class
    the time system was set
    //the cuda parallel train was set

    '''

    def __init__(self, NameID=None, index=None, cfgfile=None, basepath=None):
        super(BaseTrain, self).__init__()
        self.projectStartTime = None
        # self.initFinish = False
        self.cfgfile = cfgfile
        self.basepath = basepath

        self._argparserinit()

        if self.args.NameID == None:
            if NameID != None:
                self.NameID = NameID
            else:
                raise ValueError
        else:
            self.NameID = self.args.NameID

        self.index = self.args.index if self.args.index else index
        self.trainID = f"{self.NameID}_{self.index}"

        self._cfginit(cfgfile)
        self._loginit()

        self.set_module()

    def _trainInit(self):
        self._deviceinit()
        self._moduleinit()
        self.datasetinit()
        self.dataloaderinit()
        self.trainName = self.__class__.__name__
        self.netName = self.net.__class__.__name__
        self._title()
        self.loginfo()
        self.otherinit()

    def _argparserinit(self):
        parser = argparse.ArgumentParser(description="baseclass class for network training")
        parser.add_argument("-n", "--NameID", type=str,
                            default=None, help="the trainid name to train")
        parser.add_argument("-i", "--index", type=int,
                            default=None, help="the netfile index number to train")
        parser.add_argument("-m", "--message", type=str,
                            default=None, help="the additional message")
        parser.add_argument("-e", "--epoch", type=int,
                            default=None, help="number of epochs")
        parser.add_argument("-b", "--batchsize", type=int,
                            default=None, help="mini-batch size")
        parser.add_argument("-r", "--lr", type=float, default=None,
                            help="learning rate for gradient descent")
        parser.add_argument("-p", "--checkpoint", type=int,
                            default=None, help="print frequency")
        parser.add_argument("-t", "--threshold", type=float, default=None,
                            help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float,
                            default=None, help="ratio of conf and offset loss")
        parser.add_argument("-c", "--cudanum", type=int,
                            default=0, help="the number of cuda")
        parser.add_argument("-f", "--train", action="store_true",
                            default=False, help="start to train")
        parser.add_argument("--argv1", type=str,
                            default=None, help="start to train")
        parser.add_argument("--argv2", type=str,
                            default=None, help="start to train")
        parser.add_argument("--argv3", type=str,
                            default=None, help="start to train")

        parser.add_argument("--parallel", dest="parallel", action="store_true",
                            default=False, help="train in parallel mode")
        parser.add_argument("--clean", dest="clean", action="store_true",
                            default=False, help="start to train")
        self.args = parser.parse_args()

    def _cfginit(self, cfgfile):

        self.config = configparser.ConfigParser()
        self.config.read(cfgfile)

        saveDir_ = self.config.get(self.NameID, "SAVEDIR")

        self.dataDir = self.config.get(self.NameID, "DATADIR")

        self.saveDir = os.path.join(self.basepath, saveDir_)

        self.epoch = self.args.epoch if self.args.epoch else self.config.getint(self.NameID, "EPOCH")

        self.alpha = self.args.alpha if self.args.alpha else self.config.getfloat(self.NameID, "ALPHA")

        self.batchSize = self.args.batchsize if self.args.batchsize else self.config.getint(self.NameID, "BATCHSIZE")

        self.checkPoint = self.args.checkpoint if self.args.checkpoint else self.config.getint(self.NameID,
                                                                                               "CHECKPOINT")

        self.threshold = self.args.threshold if self.args.threshold else self.config.getfloat(self.NameID, "THRESHOLD")
        self.lr = self.args.lr if self.args.lr else self.config.getfloat(self.NameID, "LR")
        self.formalTrain = self.args.train
        self.parallelTrain = self.args.parallel
        self.cudaNum = self.args.cudanum
        self.subSaveDir = os.path.join(self.saveDir, f"{self.NameID}", f"{self.trainID}")
        self.subNetSaveDir = os.path.join(self.subSaveDir, "modules")
        # self.basefile = __file__
        self.message = self.args.message
        self.argv1 = self.args.argv1
        self.argv2 = self.args.argv2
        self.argv3 = self.args.argv3
        self.epochPoint = 1
        if self.args.clean:
            removepath(self.subSaveDir)
            # removepath(self.subNetSaveDir)

        if self.formalTrain:
            makedir(self.subSaveDir)
            makedir(self.subNetSaveDir)

        # self.transform = transforms.ToTensor()

        varc = {k.lower(): v for k, v in vars(self).items()}
        configitems = self.config.items(self.NameID)
        for configkey, configvalue in configitems:
            if configkey.lower() not in varc.keys():
                print(f"{configkey}={configvalue}")
                try:
                    exec(f"self.{configkey}={configvalue}")
                except:
                    exec(f"self.{configkey}='{configvalue}'")

    def _deviceinit(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cudaNum}")

    def _moduleinit(self):

        self.netfile = os.path.join(self.subSaveDir, f"{self.trainID}.pt")
        self.netfile_backup = os.path.join(self.subSaveDir, f"{self.trainID}_bak.pt")
        self.net.to(self.device)

        if os.path.exists(self.netfile):
            try:
                self.net.load_state_dict(torch.load(self.netfile, map_location=self.device))
                print("load successfully")
                print()
            except:
                if os.path.exists(self.netfile_backup):
                    try:
                        self.net.load_state_dict(torch.load(self.netfile_backup, map_location=self.device))
                        print("load successfully")
                        print()
                    except:
                        print("not load successfully")
        # if self.parallelTrain:
        #     cuda_idlist = [
        #         [0, 1, 2, 3],
        #         [1, 2, 3, 0],
        #         [2, 3, 0, 1],
        #         [3, 0, 1, 2]
        #     ]
        #     self.net = torch.nn.DataParallel(self.net, device_ids=cuda_idlist[self.cudaNum])

    def _loginit(self, test=True):
        self.logFile = os.path.join(self.subSaveDir, f"{self.trainID}.log")
        self.paramLogFile = os.path.join(self.subSaveDir, f"{self.trainID}_param.log")
        self.logTXTFile = os.path.join(self.subSaveDir, f"{self.trainID}.txt")

        self.logFile_backup = os.path.join(
            self.subSaveDir, f"{self.trainID}_bak.log")
        self.logFileTest_backup = os.path.join(
            self.subSaveDir, f"{self.trainID}_test_bak.log")
        if os.path.exists(self.logFile):
            self.logDict = torch.load(self.logFile)
        else:
            self.logDict = {"i": [], "j": [], "k": [], "loss": [], "time": []}
        if test:
            self.logFileTest = os.path.join(
                self.subSaveDir, f"{self.trainID}_test.log")
            if os.path.exists(self.logFileTest):
                self.logDictTest = torch.load(self.logFileTest)
            else:
                self.logDictTest = {"i": [], "j": [], "k": []}

    def _log(self, isTest: bool = False, **kwargs):
        if isTest:
            for key, value in kwargs.items():
                self.logDictTest[key].append(value)
            torch.save(self.logDictTest, self.logFileTest)
            torch.save(self.logDictTest, self.logFileTest_backup)
        else:
            for key, value in kwargs.items():
                self.logDict[key].append(value)
            torch.save(self.logDict, self.logFile)
            torch.save(self.logDict, self.logFile_backup)

    def _title(self):
        if not os.path.exists(self.logTXTFile):
            self.projectStartTime = self._get_time()
            title = f"{self.trainID.upper()} MODULE"
            self.printlog(f"{title}")
            self.printlog(self.net)
            title = f"{self.trainID} Docs"
            self.printlog(f"{title.upper()}")
            self.printlog(f"Net_DOC:\n\t{self.net.__doc__.strip()}", newline=True)
            self.printlog(f"Dataset_DOC:\n\t{self.dataset.__doc__.strip()}", newline=True)
            self.printlog(f"Train_DOC:\n\t{self.__doc__.strip()}", newline=True)
        starttime = self._get_time()
        # starttime = time .strftime("%b/%d/%Y %H:%M:%S", time.localtime())
        command = f"python {' '.join(sys.argv)}"
        titlestr = "\n" \
                   f"{'*' * 100}\n" \
                   f"*{self.trainID:^98s}*\n" \
                   f"*{command:^98s}*\n" \
                   f"*{starttime:^98s}*\n" \
                   f"{'*' * 100}"
        self.printlog(titlestr)

    def loginfo(self, **kwargs):




        title = f"{self.trainID} Parameters"
        self.printlog(f"{title:^50}")
        varc, diffdict = self._get_var(self.paramLogFile)

        self.printlog(diffdict, newline=False)

        if self.formalTrain:
            torch.save(varc, self.paramLogFile)

    def printlog(self, *args, newline=True):
        if self.formalTrain:
            with open(self.logTXTFile, 'a', encoding='utf-8') as file:
                for arg in args:
                    if isinstance(arg, dict):
                        argkeys = sorted(arg)
                        for key in argkeys:
                            print(f"{key:<11s}: ", end="")
                            print(arg[key])
                            print(f"{key:<11s}: ", end="", file=file)
                            print(arg[key], file=file)
                    else:
                        print(arg, end=' ')
                        print(arg, file=file, end=' ')
                if newline:
                    print(file=file)
                    print()
        else:
            for arg in args:
                if isinstance(arg, dict):
                    argkeys = sorted(arg)
                    for key in argkeys:
                        print(f"{key:<11s}: ", end="")
                        print(arg[key])
                else:
                    print(arg, end=' ')
            if newline:
                print()

    def saveBatch(self):
        if self.formalTrain:
            try:
                self._log(i=self.i, j=self.j, k=self.k, loss=self.loss.item(), time=self.trainTime)
                if self.parallelTrain:
                    torch.save(self.net.module.state_dict(), self.netfile)
                else:
                    torch.save(self.net.state_dict(), self.netfile)
            except KeyboardInterrupt:
                # t1 = time.time()
                if self.parallelTrain:
                    torch.save(self.net.module.state_dict(), self.netfile)
                    torch.save(self.net.module.state_dict(), self.netfile_backup)
                else:
                    torch.save(self.net.state_dict(), self.netfile)
                    torch.save(self.net.state_dict(), self.netfile_backup)


    def saveEpoch(self):
        if self.formalTrain:
            self.plot("loss")
            self.netfile_epoch = os.path.join(self.subNetSaveDir, f"{self.NameID}_{self.index}_{self.i}.pt")
            if self.parallelTrain:
                torch.save(self.net.module.state_dict(), self.netfile_epoch)
            else:
                torch.save(self.net.state_dict(), self.netfile_epoch)

    def onehot(self, a, cls=2):
        b = torch.zeros(a.size(0), cls).scatter_(-1, a.view(-1, 1).long(), 1).to(self.device)
        return b

    def plot(self, *args, isTest=False):
        for item in args:
            plotName = f"plot_{self.trainID}_{item}.png"  # jpg in linux wrong
            plotPath = os.path.join(self.subSaveDir, plotName)
            if isTest:
                y = np.array(self.logDictTest[item])
            else:
                y = np.array(self.logDict[item])
            plt.clf()
            plt.title(item)
            plt.plot(y)
            plt.savefig(plotPath)

    def train(self):
        # print(1 / 0)
        self._trainInit()
        print("train start")
        if self.parallelTrain:
            cuda_idlist = [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2]
            ]
            self.net = torch.nn.DataParallel(self.net, device_ids=cuda_idlist[self.cudaNum])
        self.dataloader = data.DataLoader(self.dataset, self.batchSize, shuffle=True)
        self.dataloaderLen = len(self.dataloader)
        self.i = 0
        self.j = 0
        self.k = 0
        self.trainTime = 0
        if self.projectStartTime == None:
            self.projectStartTime = self._get_time()

        if len(self.logDict["i"]) > 0:
            self.i = self.logDict["i"][-1]
            self.j = self.logDict["j"][-1] + 1
            self.k = self.logDict["k"][-1]
            self.trainTime = self.logDict["time"][-1]
            if self.j >= self.dataloaderLen:
                self.i += 1
                self.j = 0
        self.trainImpl()

    # @abc.abstractmethod
    def trainImpl(self):

        while (self.i < self.epoch):
            self.net.train()
            self.net.to(self.device)
            for dataitem in self.dataloader:
                startTime = time.time()
                output = self.get_output(dataitem)

                self.loss = self.get_loss(output)

                self.step()
                endTime = time.time()
                self.batchTime = endTime - startTime
                self.trainTime += self.batchTime
                if 0 == (self.j + 1) % self.checkPoint or self.j == self.dataloaderLen - 1:
                    self.accuracy = self.get_accuracy(output)
                    result = self.get_result()
                    self.printlog(result, newline=True)
                    self.saveBatch()

                self.j += 1
                self.k += 1

                if self.j == self.dataloaderLen:
                    self.j = 0
                    break
            if 0 == (self.i + 1) % self.epochPoint:
                self.saveEpoch()

            self.test()

            self.i += 1

    def step(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def test(self):

        with torch.no_grad():
            self.net.eval()
            self.testImpl()

    def testImpl(self):
        pass
        # for dataitem in self.testdataloader:
        #     output = self.get_output(dataitem)
        #     self.accuracy = self.get_accuracy(output)
        #     result = f"test:{self.get_result()}"
        #     self.printlog(result, newline=True)
        # break

    @property
    def optimizer(self, *args, **kwargs):
        raise NotImplementedError
        # return eval(self.optimDict[self.optimid])

    @property
    def lossFn(self, *args, **kwargs):
        raise NotImplementedError
        # return eval(self.lossDict[self.lossid])

    @abc.abstractmethod
    def set_module(self, *args, **kwargs):
        self.net = BaseNet(*args, **kwargs)

    @abc.abstractmethod
    def datasetinit(self):
        self.dataset = []
        self.testdataset = []
        if 0 == len(self.dataset):
            print("dataset must be initialed")
            exit(-1)

    def dataloaderinit(self):
        self.dataloader = data.DataLoader(self.dataset, self.batchSize, shuffle=True)
        # self.testdataloader = data.DataLoader(self.testdataset, self.batchSize, shuffle=True)

    def get_result(self, *args, **kwargs):
        result = f"epoch: {self.i:>4d}, batch: {self.j:>4d}, totalbatch: {self.k:>6d}, loss: {self.loss.item():.4f}, " \
                 f"batchtime: {self.batchTime}, traintime: {self.trainTime}"
        return result

    @abc.abstractmethod
    def get_loss(self, *args, **kwargs):
        loss = torch.Tensor([])
        if 0 == loss.size(0):
            print("get_loss function should be overrided")
            exit(-1)
        return loss

    @abc.abstractmethod
    def get_output(self, *args, **kwargs):
        output = torch.Tensor([])
        if 0 == output.size(0):
            print("get_loss function should be overrided")
            exit(-1)
        return output

    # @abc.abstractmethod
    def get_accuracy(self, *args, **kwargs):
        pass
        # accuracy = torch.Tensor([])
        # # if 0 == accuracy.size(0):
        # #     print("get_accuracy function should be overrided")
        # #     exit(-1)
        # return accuracy

    # @abc.abstractmethod
    def get_property(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def get_precision(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def get_recall(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def detect(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def analyze(self, *args, **kwargs):
        pass

    def otherinit(self):
        pass

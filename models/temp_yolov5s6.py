import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # if 1:
            # print(f"x:{x.shape}") # list

        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            if 0:
                print(f"x[i]:{x[i].shape}")
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                if 0:
                    print(f"x[i]:{x[i].shape}")
                    print(f"self.grid[i].shape:{self.grid[i].shape}")
                    print(f"self.anchor_grid[i]:{self.anchor_grid[i].shape}\n{self.anchor_grid[i]}") # [1, 3, 1, 1, 2]
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
        # return x if self.training else [torch.cat(z, 1), x]

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float() # x在前与cx在前一致


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
import time
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import re
import yaml
size = 640
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../yolov5l.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[size, size], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
    if 1:
        print(type(y))
        if isinstance(y, tuple) or isinstance(y, list):
            for y_ in y:
                if isinstance(y_, tuple) or isinstance(y_, list):
                    print(type(y_))
                    for _y in y_:
                        if isinstance(_y, torch.Tensor):
                            print(_y.shape)
                else:
                    if isinstance(y_, torch.Tensor):
                        print(y_.shape)
        else:
            if isinstance(y, torch.Tensor):
                print(y.shape)
    # print("-----------parameter--------------------------")
    # for n, p in model.named_parameters():
    #     print(n)
    #     print(p.shape)
    print("-----------module--------------------------")
    j = 0
    modeldict = {}
    modellist = []
    # for i, (n, p) in enumerate(model.named_modules()):
    #     print(n)
    #     print(type(p))
    # exit()
    # model.2.m.2.cv1.conv
    aset = set()
    bset = set()
    bn = 0
    bottlelist = []
    typeset = set()
    sizelist = []
    size = 640
    for i, (n, p) in enumerate(model.named_modules()):
        # print(f"{i=}")
        # print(f"{n=}")
        # print(p)
        # print(type(p))
        tpe = type(p)

        print(f"{tpe=}, {n=}")
        ns = re.match("model\.(\d+)$", n)
        # ns2 = re.match("model\.(\d+)\.m\.(\d+)\.[^0-9]+\w+$", n)
        # if ns2:
        #     print(f"{tpe=},  {n=}, {ns2.group(1)=}, {ns2.group(2)=}")

        # exit()
        tp = str(type(p))
        cp = re.compile("models\.(\w+)\.(\w+)")
        up = re.compile("torch\.nn\.modules\.upsampling\.Upsample")
        s = cp.search(tp)
        u = up.search(tp)
        if s:
            t = s.group(2)
        # if s:
        #     print(f"{s.group(2)=}")
        if t == "Bottleneck":
            # bn += 1
            bottlelist.append('1')

        if ns:
            # print(f"{ns.group()=}")
            k = int(ns.group(1))
            print(f"{k=}")
            typeset.add(t)
            # tpe = type(p)
            # print(f"{tpe=}")
            # print(f"{hasattr(p, 'cv1')=}")
            # print(f"{dir(p)=}")
            # if hasattr(p, "cv1") or hasattr(p,"conv"):
            #     for n1, p1 in p.named_parameters():
            #         print(n1)
            #         print(p1.shape)
            if u:
                v = [-1, 1, 'nn.Upsample', [None, 2, 'nearest']]
                modeldict[k] = v
                size *= 2
            if s:

                # print(f"{j=}")
                # print(f"{s.group(1)=}, {s.group(2)=}")
                # print(n)
                # print(p)

                modellist.append(t)
                if t in ["Focus"]:
                    size //= 2

                    for n1, p1 in p.named_parameters():
                        # print(n1)
                        # print(p1.shape)
                        v = [-1, p1.shape[0], t, [64, 3]]
                        modeldict[k] = v
                        break
                    # exit(0)
                elif t in ["Conv"]:

                    # print(f"{t=}")
                    for n2, p2 in p.named_modules():
                        # print(n2)
                        # print(isinstance(p2,nn.Conv2d))
                        if isinstance(p2,nn.Conv2d):
                            # print(p2.kernel_size, p2.stride)
                            # print(p2.in_channels, p2.out_channels)
                            # k2 = p2.kernel_size[0]
                            # s2 = p2.stride[0]
                            if p2.kernel_size[0]>1:
                                size //= 2
                            v = [-1, 1, t, [p2.out_channels, p2.kernel_size[0], p2.stride[0]]]
                            print(f"{v=}")
                            # modellist.append(v)
                            modeldict[k] = v
                            break
                    # for n1, p1 in p.named_parameters():
                    #     print(n1)
                    #     print(p1.shape)
                    #     # print(f"{p.kernel=}")
                    #     if re.match("[\w.]+bias", n1):
                    #
                    #         v = [-1, 1, t, [p1.shape[0], p1.shape[0], 2]]
                    #         print(f"{v=}")
                    #         # modellist.append(v)
                    #         modeldict[k] = v
                    #         break
                    #     exit(0)
                elif t in ["C3"]:
                    flag = True
                    print(f"{t=}")
                    # print(p.parameters)
                    for n1, p1 in p.named_parameters():
                        # print()
                        # print(p1.shape)
                        # print(p)
                        if re.match("[\w.]+weight", n1):
                            v = [-1, t, [p1.shape[1]]]
                            # print(f"{v=}")
                            # modellist.append(v)
                            modeldict[k] = v
                            bottlelist.append('0')
                            # bn = 0
                            break
                elif t in ["SPP"]:
                    pass
                elif t in ["Concat"]:
                    pass
                elif t in ["Detect"]:
                    pass
                # elif t in ["Bottleneck"]:
                #     print(f"{t=}")
                    # exit(0)
                    # exit()
                # modeldict[k] = t
                sizelist.append(size)
                j+=1

            # print(type(p))
            # tp = str(type(p))
        # else:
        #     print(f"{s is None=}") # True
            print("========")

    print("-----------children------------------------")
    # for n, p in model.named_children():
    #     print(n)
    print("--------------------")
    print(modeldict)
    # print(modellist)
    # with open("yolov5s6.yaml", "w") as f:
    #     yaml.dump(modeldict, stream=f)
    # focus = Focus(1,2)
    # print(type(focus))
    # print(dir(focus))
    # print(hasattr(focus, "conv"))

    print(modellist)
    exit()
    modellist2 = filter(lambda x:x=="C3" or x=="Bottleneck", modellist)
    print(list(modellist2))
    print(list(modeldict.values()))
    print(bottlelist)
    bottlelist2 = [len(x) for x in "".join(bottlelist).split("0") if len(x)>0]
    print(bottlelist2)
    i = 0
    for key, value in modeldict.items():
        print(key)
        print(value)
        if re.search("C3", str(value)):
            value.insert(1, bottlelist2[i])
            i += 1
    print(modeldict)
    print(typeset)
    print(sizelist)
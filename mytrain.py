#!/e/Soft/Python/Python38/python

# !/usr/bin/python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# @File Name:        mytrain.py.py
# @Author:           wen
# @Version:          ver0_1
# @Created:          2021/2/19 10:30
# @Description:      Main Function:    xxx
# Function List:     hello() -- print helloworld
# History:
#       <author>    <version>   <time>      <desc>
#       wen         ver0_1      2020/12/15  xxx
# ------------------------------------------------------------------
import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mytest as test # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.myyolo import Model
from utils.autoanchor import check_anchors
from utils.mydatasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging
from utils.google_utils import attempt_download
from utils.myloss import compute_loss
# from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f'Hyperparameters {hyp}')
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    if True:
        plots = False
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check

    train_path = data_dict['train']
    test_path = data_dict['val']

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt') and os.path.exists(weights)
    # pretrained = False
    if pretrained:
        # with torch_distributed_zero_first(rank):
        #     attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = Model(opt.cfg or ckpt['model'].yaml, ch=4, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=4, nc=nc).to(device)  # create

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loggers = {'wandb': wandb}  # loggers dict

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    if 0:
        print(f"imgsz:{imgsz}, imgsz_test:{imgsz_test}")
        exit(1108)
    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers, pad=0.5)[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), opt.warmup_epoch)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs))


    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            # else:
            #     print("\nWarmup finished")
            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            if 0:
                print(f"imgs:{imgs.shape}")
                exit(1046)
            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 0:
                    for y in pred:
                        if isinstance(y, torch.Tensor):
                            print(y.shape)
                        elif isinstance(y, list):
                            for y_ in y:
                                print(y_.shape)
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                if 1:
                    print(f"s--------------------:{s}") # lbox, lobj, lcls, loss
                    # 0/29     3.68G   0.09294    0.0258   0.04462    0.1634        15       640
                pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            checkpoint = opt.check_point
            if 0:
                print(f"epoch % checkpoint:{epoch % checkpoint} {epoch}")
            if (epoch+1) % checkpoint == 0 and not opt.notest or final_epoch:  # Calculate mAP
                # if False:  # Calculate mAP
                if False:
                    # RuntimeError: Sizes of tensors must match except in dimension 2. Got 11 and 12 (The offending index is 0)
                    results, maps, times = test.test(opt.data,
                                                     batch_size=total_batch_size,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema,
                                                     single_cls=opt.single_cls,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     plots=plots and final_epoch,
                                                     log_imgs=opt.log_imgs if wandb else 0)

                    # Write
                    with open(results_file, 'a') as f:
                        f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    if fi > best_fitness:
                        best_fitness = fi

                    # Save model
                    save = (not opt.nosave) or (final_epoch and not opt.evolve)
                    if save:
                        with open(results_file, 'r') as f:  # create checkpoint
                            ckpt = {'epoch': epoch,
                                    'best_fitness': best_fitness,
                                    'training_results': f.read(),
                                    'model': ema.ema,
                                    'optimizer': None if final_epoch else optimizer.state_dict(),
                                    'wandb_id': None}
                        if True:
                            epochsave_file = wdir / rf"{epoch}.pt"
                            torch.save(ckpt, epochsave_file)
                            # print(f"epoch:{ckpt['epoch']}")
                        # Save last, best and delete
                        torch.save(ckpt, last)
                        if best_fitness == fi:
                            torch.save(ckpt, best)
                            if True:
                                print(f"epoch best: {ckpt['epoch']}")
                                bestsave_file = wdir / rf"best_{epoch}.pt"
                                torch.save(ckpt, bestsave_file)

                        del ckpt
                else:
                    try:
                        results, maps, times = test.test(opt.data,
                                                         batch_size=total_batch_size,
                                                         imgsz=imgsz_test,
                                                         model=ema.ema,
                                                         single_cls=opt.single_cls,
                                                         dataloader=testloader,
                                                         save_dir=save_dir,
                                                         plots=plots and final_epoch,
                                                         log_imgs=opt.log_imgs if wandb else 0)
                        if 1:
                            print(f"result================:{results}")
                            # 0.0, 0.0, 0.0, 0.0, 0.0943969115614891, 0.044368892908096313, 0.046762723475694656
                            exit(1023)
                        # Write
                        with open(results_file, 'a') as f:
                            f.write(
                                s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

                        # Update best mAP
                        fi = fitness(
                            np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                        if fi > best_fitness:
                            best_fitness = fi

                        # Save model
                        save = (not opt.nosave) or (final_epoch and not opt.evolve)
                        if save:
                            with open(results_file, 'r') as f:  # create checkpoint
                                ckpt = {'epoch': epoch,
                                        'best_fitness': best_fitness,
                                        'training_results': f.read(),
                                        'model': ema.ema,
                                        'optimizer': None if final_epoch else optimizer.state_dict(),
                                        'wandb_id': None}
                            if True:
                                epochsave_file = wdir / rf"epoch_{epoch}.pt"
                                torch.save(ckpt, epochsave_file)
                                # print(f"epoch:{ckpt['epoch']}")
                            # Save last, best and delete
                            torch.save(ckpt, last)
                            if best_fitness == fi:
                                torch.save(ckpt, best)
                                if True:
                                    # print(f"epoch best: {ckpt['epoch']}")
                                    bestsave_file = wdir / rf"best_{epoch}.pt"
                                    torch.save(ckpt, bestsave_file)

                            del ckpt


                    except RuntimeError as e:
                        print(logger.info(e))
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    # exit(1120)
    if rank in [-1, 0]:
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)  # strip optimizers

        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    else:
        dist.destroy_process_group()

    # wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    torch.set_printoptions(precision=3, threshold=100000, sci_mode=False)
    from tool.utils import record_log, get_time
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=r'models/myyolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/woodstest.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='G:/GSave/yolov5_test/runs/train', help='save to project/name')
    parser.add_argument('--name', default='test', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--time', type=str, default=get_time(), help='project run start time')
    parser.add_argument('--warmup-epoch', type=int, default=50, help='epochs of warmup')
    parser.add_argument('--check-point', type=int, default=1, help='checkpoint')

    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    if True:
        record_log(opt.save_dir)
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device)

import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.nn import DataParallel

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchbiomed.datasets as dset
import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss
import torchbiomed.utils as utils

import os
import sys
import math

import shutil

# import setproctitle

import vnet
# import make_graph
from functools import reduce
import operator


nodule_masks = "normalized_nodule_masks"
lung_masks = "normalized_lung_masks"
ct_images = "normalized_ct_images"
# ct_targets = nodule_masks
ct_targets = lung_masks

# target_split = [2, 2, 2]
# target_split = [4, 4, 4]

# 64x64x64 mm3
target_split = [4, 5, 5]
# target_split = [5, 6, 6]
# target_split = [5, 5, 5]
# target_split = [5, 4, 4]


def inference(args, loader, model, transforms):
    src = args.inference
    dst = args.save

    model.eval()
    nvols = reduce(operator.mul, target_split, 1)
    # assume single GPU / batch size 1
    for data in loader:
        # we infere EXACTLY one data per-gpu
        data, series, origin, spacing = data[0]
        shape = data.size()
        print "data shape {0}".format(shape)
        # convert names to batch tensor
        if args.cuda:
            # data.pin_memory()
            data = data.cuda()
        data = Variable(data, volatile=True)
        try:
            output = model(data)
        except Exception, e:
            print e
            return

        print "output data shape {0}".format(output.size())
        _, output = output.max(1)
        output = output.view(shape)
        output = output.cpu()
        # merge subvolumes and save
        results = output.chunk(nvols)
        results = map(lambda var: torch.squeeze(var.data).numpy().astype(np.int16), results)
        # results = map(lambda var: var.transpose([1, 2, 0]), results)
        image_size = shape[-3:]
        # transform torch.ToTensor altered results back
        image_size = [image_size[1], image_size[2], image_size[0]]
        new_results = []
        for img in results:
            img = img.reshape(image_size)
            new_results.append(img)
        results = new_results
        volume = utils.merge_image(results, target_split)

        # volume = np.zeros([128, 128, 128])
        # z_p, y_p, x_p = 2, 2, 2
        # z_incr, y_incr, x_incr = 64, 64, 64
        # idx = 0
        # for zi in range(z_p):
        #     zstart = zi * z_incr
        #     zend = zstart + z_incr
        #     for yi in range(y_p):
        #         ystart = yi * y_incr
        #         yend = ystart + y_incr
        #         for xi in range(x_p):
        #             xstart = xi * x_incr
        #             xend = xstart + x_incr
        #             volume[zstart:zend, ystart:yend, xstart:xend] = results[idx]
        #             idx +=1
        print("save {}".format(series))
        utils.save_updated_image(volume, os.path.join(dst, series + ".mhd"), origin, spacing)


def inference_piecebypiece(args, loader, model, transforms):
    src = args.inference
    dst = args.save

    model.eval()
    nvols = reduce(operator.mul, target_split, 1)
    # assume single GPU / batch size 1
    for loaded_data in loader:
        batch_data, series, origin, spacing = loaded_data[0]
        shape = batch_data.size()

        batches = shape[0]
        image_size = shape[-3:]

        results = []
        first_output = True
        batch_output = []
        batch_interval = 1
        for batch in range(batches):
            print "processing batch {0}".format(batch)

            if not batch % batch_interval == 0:
                continue

            data = batch_data[batch:batch + batch_interval, ...]
            # convert names to batch tensor
            if args.cuda:
                # data.pin_memory()
                data = data.cuda()
            data = Variable(data, volatile=True)
            # data = data.unsqueeze(0)
            try:
                output = model(data)
            except Exception, e:
                print e
                return
            _, output = output.max(1)

            batch_shape = [batch_interval] + [dim for dim in shape[1:]]
            output = output.view(batch_shape)
            output = output.cpu()
            # output = output.unsqueeze(0)

            if first_output:
                batch_output = output
                first_output = False
            else:
                batch_output = torch.cat([batch_output, output])


        print "batch_output size {0}".format(batch_output.size())
        results = batch_output.chunk(nvols)
        results = map(lambda var: torch.squeeze(var.data).numpy().astype(np.int16), results)
        image_size = [image_size[1], image_size[2], image_size[0]]
        new_results = []
        for img in results:
            img = img.reshape(image_size)
            new_results.append(img)
        results = new_results
        volume = utils.merge_image(results, target_split)

        print("save {}".format(series))
        utils.save_updated_image(volume, os.path.join(dst, series + ".mhd"), origin, spacing)

# performing post-train inference:
# train.py --resume <model checkpoint> --i <input directory (*.mhd)> --save <output directory>

def noop(x):
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dice', default=False, action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--infertype', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', default=False, action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='rmsprop',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.cuda:
        print 'no cuda support!\n'

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=False)

    if args.cuda:
        gpu_ids = range(args.ngpu)
        model = DataParallel(model, device_ids=gpu_ids)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save)

    # LUNA16 dataset isotropically scaled to 2.5mm^3
    # and then truncated or zero-padded to 160x128x160
    normMu = [-510.154]
    normSigma = [474.620]
    normTransform = transforms.Normalize(normMu, normSigma)

    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 1} if args.cuda else {}
        src = args.inference
        dst = args.save

        # we infere EXACTLY one data per-gpu!
        # inference_batch_size = args.ngpu
        inference_batch_size = 1
        root = os.path.dirname(src)
        images = os.path.basename(src)
        print "inferencing root: {0}, img:{1}".format(root, images)
        dataset = dset.LUNA16(root=root, images=images, transform=trainTransform, split=target_split, mode="infer")
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        if args.infertype == 0:
            inference(args, loader, model, trainTransform)
        elif args.infertype == 1:
            inference_piecebypiece(args, loader, model, trainTransform)
        else:
            print "inference type wrong"
        return


if __name__ == '__main__':
    main()

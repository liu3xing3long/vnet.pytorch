#!/usr/bin/env python3

import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchbiomed.datasets as dset
import torchbiomed.loss as bioloss
import torchbiomed.utils as utils
import os
import shutil

import vnet
from functools import reduce
import operator
import re

#nodule_masks = "normalized_mask_5_0"
#lung_masks = "normalized_seg_lungs_5_0"
#ct_images = "normalized_CT_5_0"
#target_split = [1, 1, 1]
#ct_targets = nodule_masks


nodule_masks = "normalized_nodule_masks"
lung_masks = "normalized_lung_masks"
ct_images = "normalized_ct_images"
# ct_targets = nodule_masks
ct_targets = lung_masks

# target_split = [2, 2, 2]
target_split = [4, 4, 4]

# 64x64x64 mm3
# target_split = [4, 5, 5]
# target_split = [5, 5, 5]
data_set_path = "luna16_1mm_xyz"


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def inference(args, loader, model, transforms):
    src = args.inference
    dst = args.save

    model.eval()
    nvols = reduce(operator.mul, target_split, 1)
    # assume single GPU / batch size 1
    for data in loader:
        data, series, origin, spacing = data[0]
        shape = data.size()
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        data = Variable(data, volatile=True)
        try:
            output = model(data)
        except Exception, e:
            print e
            return

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

# performing post-train inference:
# train.py --resume <model checkpoint> --i <input directory (*.mhd)> --save <output directory>

def noop(x):
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dice', default=False, action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
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
    best_prec1 = 100.
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.cuda:
        print 'no cuda support!\n'

    tar_split_str = str(target_split)
    re.sub('\s', '', tar_split_str)

    args.save = args.save or 'work/{0}_{1}_{2}'.format(data_set_path, tar_split_str, datestr())
    nll = True
    if args.dice:
        nll = False
    weight_decay = args.weight_decay

    # setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=nll)

    if args.cuda:
        batch_size = args.ngpu*args.batchSz
        gpu_ids = range(args.ngpu)
        model = nn.parallel.DataParallel(model, device_ids=gpu_ids)
    else:
        batch_size = args.batchSz

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
    else:
        model.apply(weights_init)

    if nll:
        train = train_nll
        test = test_nll
        class_balance = True
    else:
        train = train_dice
        test = test_dice
        class_balance = False

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
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    if ct_targets == nodule_masks:
        masks = lung_masks
    else:
        masks = None

    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 1} if args.cuda else {}
        src = args.inference
        dst = args.save
        inference_batch_size = args.ngpu
        root = os.path.dirname(src)
        images = os.path.basename(src)
        print "inferencing root: {0}, img:{1}".format(root, images)
        dataset = dset.LUNA16(root=root, images=images, transform=testTransform, split=target_split, mode="infer")
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        inference(args, loader, model, trainTransform)
        return

    kwargs = {'num_workers': 2, 'pin_memory': False} if args.cuda else {}
    print("loading training set")
    trainSet = dset.LUNA16(root=data_set_path, images=ct_images, targets=ct_targets,
                           mode="train", transform=trainTransform,
                           class_balance=class_balance, split=target_split, seed=args.seed, masks=masks)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)

    print("loading test set")
    testSet = dset.LUNA16(root=data_set_path, images=ct_images, targets=ct_targets,
                          mode="test", transform=testTransform,
                          seed=args.seed, masks=masks, split=target_split)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, **kwargs)

    target_mean = trainSet.target_mean()
    bg_weight = target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    print(bg_weight)
    class_weights = torch.FloatTensor([bg_weight, fg_weight])
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    err_best = 100.
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, trainF, class_weights)
        err = test(args, epoch, model, testLoader, optimizer, testF, class_weights)
        is_best = False
        if err < best_prec1:
            is_best = True
            best_prec1 = err
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.save, "vnet")
        # os.system('./plot.py {} {} &'.format(len(trainLoader), args.save))

    trainF.close()
    testF.close()


def train_nll(args, epoch, model, trainLoader, optimizer, trainF, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target = target.view(target.numel())
        loss = F.nll_loss(output, target, weight=weights)
        dice_loss = bioloss.dice_error(output, target)
        # make_graph.save(loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}\t Dice: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err, dice_loss))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test_nll(args, epoch, model, testLoader, optimizer, testF, weights):
    model.eval()
    test_loss = 0
    dice_loss = 0
    incorrect = 0
    numel = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.view(target.numel())
        numel += target.numel()
        output = model(data)
        test_loss += F.nll_loss(output, target, weight=weights).data[0]
        dice_loss += bioloss.dice_error(output, target)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    dice_loss /= len(testLoader)
    err = 100.*incorrect/numel
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
        test_loss, incorrect, numel, err, dice_loss))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err


def train_dice(args, epoch, model, trainLoader, optimizer, trainF, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = bioloss.dice_loss(output, target)
        # make_graph.save(loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        err = 100.*(1. - loss.data[0])
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tError: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test_dice(args, epoch, model, testLoader, optimizer, testF, weights):
    model.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = bioloss.dice_loss(output, target).data[0]
        test_loss += loss
        incorrect += (1. - loss)

    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average Dice Coeff: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()

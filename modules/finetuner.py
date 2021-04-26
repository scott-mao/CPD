import os
import time, datetime

import torch
import torch.nn as nn

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56, resnet_110
from models.cifar10.googlenet import googlenet, Inception
from models.cifar10.densenet import densenet_40
from models.load_model import load_vgg_model, load_google_model, load_resnet_model, load_densenet_model

from collections import OrderedDict
from data import cifar10, imagenet
from thop import profile
import utils.common as utils

class Finetuner:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger

        # load training data
        self.input_image_size = {
            "cifar10": 32,  # CIFAR-10
            "imagenet": 224,  # ImageNet
        }[self.args.dataset]

        if self.args.dataset == 'cifar10':
            self.train_loader, self.val_loader = cifar10.load_data(self.args)
        else:
            data_tmp = imagenet.Data(args)
            self.train_loader = data_tmp.train_loader
            self.val_loader = data_tmp.test_loader

        #Cifar-10
        self.criterion = nn.CrossEntropyLoss().cuda()
        #ImageNet
        CLASSES = 1000 #label_smooth: 0.1
        self.criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, 0.1).cuda()

        self.load_model(self.get_prune_ratio())

    def load_model(self, compress_rate):
        # load model (나중에 지우자@@@@@@@@@@@)
        self.logger.info('compress_rate:' + str(compress_rate))
        self.logger.info('==> Building model..')
        self.model = eval(self.args.arch)(compress_rate=compress_rate).cuda()
        self.logger.info(self.model)

        # compute flops and params
        input_image = torch.randn(1, 3, self.input_image_size, self.input_image_size).cuda()
        flops, params = profile(self.model, inputs=(input_image,))
        self.logger.info('Params: %.2f' % (params))
        self.logger.info('Flops: %.2f' % (flops))

        if self.args.test_only:
            if os.path.isfile(self.args.test_model_dir):
                self.logger.info('loading checkpoint {} ..........'.format(self.args.test_model_dir))
                checkpoint = torch.load(self.args.test_model_dir)
                self.model.load_state_dict(checkpoint['state_dict'])
                valid_obj, valid_top1_acc, valid_top5_acc = self.validate(0, self.val_loader, self.model, self.criterion,
                                                                          self.args)
            else:
                self.logger.info('please specify a checkpoint file')
            return

        if len(self.args.device_ids) > 1:
            device_id = []
            for i in range(len(self.args.device_ids)):
                device_id.append(i)
            self.model = nn.DataParallel(self.model, device_ids=device_id).cuda()

        # load the checkpoint if it exists
        checkpoint_dir = os.path.join(self.args.job_dir, 'checkpoint.pth.tar')
        if self.args.resume:
            self.logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
            checkpoint = torch.load(checkpoint_dir)
            start_epoch = checkpoint['epoch'] + 1
            best_top1_acc = checkpoint['best_top1_acc']

            # deal with the single-multi GPU problem
            new_state_dict = OrderedDict()
            tmp_ckpt = checkpoint['state_dict']
            if len(self.args.gpu) > 1:
                for k, v in tmp_ckpt.items():
                    new_state_dict['module.' + k.replace('module.', '')] = v
            else:
                for k, v in tmp_ckpt.items():
                    new_state_dict[k.replace('module.', '')] = v

            self.model.load_state_dict(new_state_dict)
            self.logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))
        else:
            if self.args.use_pretrain:
                self.logger.info('resuming from pretrain model')
                origin_model = eval(self.args.arch)(compress_rate=[0.] * 100).cuda()
                ckpt = torch.load(self.args.pretrain_dir, map_location=self.args.device)

                # if args.arch=='resnet_56':
                #    origin_model.load_state_dict(ckpt['state_dict'],strict=False)
                if self.args.arch == 'densenet_40' or self.args.arch == 'resnet_110':
                    new_state_dict = OrderedDict()
                    for k, v in ckpt['state_dict'].items():
                        new_state_dict[k.replace('module.', '')] = v
                    origin_model.load_state_dict(new_state_dict)
                else:
                    origin_model.load_state_dict(ckpt['state_dict'])

                oristate_dict = origin_model.state_dict()

                # model: already compressed model, oristate_dict: parameters of the full model
                if self.args.arch == 'googlenet':
                    load_google_model(self.args, self.logger, self.model, oristate_dict)
                elif self.args.arch == 'vgg_16_bn':
                    load_vgg_model(self.args, self.logger, self.model, oristate_dict)
                elif self.args.arch == 'resnet_56':
                    load_resnet_model(self.args, self.logger, self.model, oristate_dict, 56)
                elif self.args.arch == 'resnet_110':
                    load_resnet_model(self.args, self.logger, self.model, oristate_dict, 110)
                elif self.args.arch == 'densenet_40':
                    load_densenet_model(self.args, self.logger, self.model, oristate_dict)
                else:
                    raise
            else:
                self.logger.info('training from scratch')

    def get_prune_ratio(self):
        if self.args.compress_rate:
            import re
            cprate_str = self.args.compress_rate
            cprate_str_list = cprate_str.split('+')
            pat_cprate = re.compile(r'\d+\.\d*')
            pat_num = re.compile(r'\*\d+')
            cprate = []
            for x in cprate_str_list:
                num = 1
                find_num = re.findall(pat_num, x)
                if find_num:
                    assert len(find_num) == 1
                    num = int(find_num[0].replace('*', ''))
                find_cprate = re.findall(pat_cprate, x)
                assert len(find_cprate) == 1
                cprate += [float(find_cprate[0])] * num

            compress_rate = cprate
        else:
            default_cprate = {
                'vgg_16_bn': [0.4, 0.5, 0.6, 0.5, 0.7, 0.6, 0.6, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9], # Aimnet 에서 돌린결과 (from 경환)
                'densenet_40': [0.0] + [0.1] * 6 + [0.7] * 6 + [0.0] + [0.1] * 6 + [0.7] * 6 + [0.0] + [0.1] * 6 + [0.7] * 5 + [0.0],  # Hrank 꺼
                'googlenet': [0.1] + [0.75] + [0.7] + [0.75] + [0.85] + [0.9] + [0.9] + [0.9] + [0.8] + [0.9], # Aimet으로 돌린
                'resnet_56': [0.] + [0.15] * 2 + [0.4] * 27,  # Hrankplus꺼
                # 'resnet_56': [0,
                #              0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1,
                #              0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1,
                #              0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1], #우리꺼 2: 슬기가 새로뽑은 것
                'resnet_110': [0.0] + [0.1] * 4 + [0.5] * 31 + [0.1] * 5 + [0.6] * 31 + [0.1] * 5 + [0.6] * 31 + [0.0], # 우리꺼 2: 슬기가 새로뽑은 것
                'resnet_50': [0.1,
                              0.1, 0.3, 0.5, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                              0.2, 0.5, 0.6, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.95, 0.95, 0.95,
                              0.2, 0.5, 0.6, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
                              0.95, 0.95, 0.95,
                              0.2, 0.5, 0.6, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.7]

            }
            compress_rate = default_cprate[self.args.arch]

            return compress_rate

    def prune(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        if self.args.dataset == 'cifar10':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min = 0.0004)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_decay_step, gamma=0.1)

        start_epoch = 0
        best_top1_acc = 0
        best_top5_acc = 0
        # adjust the learning rate according to the checkpoint
        for epoch in range(start_epoch):
            scheduler.step()

        # train the model
        epoch = start_epoch
        while epoch < self.args.epochs:
            train_obj, train_top1_acc, train_top5_acc = self.train(epoch, self.train_loader, self.model, self.criterion, optimizer, scheduler)
            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(epoch, self.val_loader, self.model, self.criterion)

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, self.args.job_dir)

            epoch += 1
            self.logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))  #

    def train(self, epoch, train_loader, model, criterion, optimizer, scheduler):
        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        model.train()
        end = time.time()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        self.logger.info('learning_rate: ' + str(cur_lr))

        num_iter = len(train_loader)
        for i, (images, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            # compute outputy
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)   #accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                self.logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Loss {loss.avg:.4f} '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, i, num_iter, loss=losses,
                        top1=top1, top5=top5))

        scheduler.step()

        return losses.avg, top1.avg, top5.avg

    def validate(self, epoch, val_loader, model, criterion):
        batch_time = utils.AverageMeter('Time', ':6.3f')
        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, target)

                # measure accuracy and record loss
                pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            self.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def cluster(self):
        self.cluster_mask = []

    def compress_model(self, mask_cluster, mask_pruning):
        mask = mask_cluster and mask_pruning

    def decompose(self):
        self.model = []


"""
DNN base trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from spars_optim import SparseSGD
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, List
from utils.utils import accuracy, AverageMeter, print_table, lr_schedule, convert_secs2time


class BaseTrainer(object):
    def __init__(self,
        model: nn.Module,
        loss_type: str, 
        trainloader, 
        validloader,
        args,
        logger,
    ):
        # model architecture
        self.model = model
        self.args = args

        # loader
        self.trainloader = trainloader
        self.validloader = validloader
        
        # loss func
        if loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type")
        
        # optimizer
        # self.optimizer = SparseSGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, 
        #         weight_decay=args.weight_decay, g_batch=args.batch_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay)
        
        if args.use_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # logger
        self.logger = logger
        self.logger_dict = {}

        # wandb logger
        if args.wandb:
            self.wandb_logger = wandb.init(entity=args.entity, project=args.project, name=args.name, config={"lr":args.lr})
            self.wandb_logger.watch(model, criterion=self.criterion, log_freq=1)
            self.wandb_logger.config.update(args)

        # learning rate scheduler
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[lr_schedule])

    def _init_scaler(self):
        """Initialize low preiciosn training
        """
        self.scaler = torch.cuda.amp.GradScaler() 

    def base_forward(self, inputs, target):
        """Foward pass of NN
        """
        if self.args.mixed_prec:
            with torch.cuda.amp.autocast():
                out = self.model(inputs)
                loss = self.criterion(out, target)
        else:
            out = self.model(inputs)
            loss = self.criterion(out, target)
        return out, loss

    def base_backward(self, loss):
        # loss = loss.mean()
        # zero grad
        self.optimizer.zero_grad()

        # backward
        if self.args.mixed_prec:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
    
    def train_step(self, inputs, target):
        """Training step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float().mul(100.)

        out, loss = self.base_forward(inputs, target)
        self.base_backward(loss)
        
        return out, loss

    def valid_step(self, inputs, target):
        """validation step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float().mul(100.)

        out, loss = self.base_forward(inputs, target)
            
        return out, loss


    def train_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()

        for idx, (inputs, target) in enumerate(self.trainloader):
            if self.args.use_cuda:
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
            
            out, loss = self.train_step(inputs, target)
            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            losses.update(loss.mean().item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        
        self.logger_dict["train_loss"] = losses.avg
        self.logger_dict["train_top1"] = top1.avg
        self.logger_dict["train_top5"] = top5.avg

    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()

        for idx, (inputs, target) in enumerate(self.validloader):
            if self.args.use_cuda:
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)

            out, loss = self.valid_step(inputs, target)

            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            losses.update(loss.mean().item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        
        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg

    def fit(self):
        self.logger.info("\nStart training: lr={}, loss={}, mixedprec={}".format(self.args.lr, self.args.loss_type, self.args.mixed_prec))

        start_time = time.time()
        epoch_time = AverageMeter()

        if self.args.mixed_prec:
            self._init_scaler()
            self.logger.info("Mixed precision scaler initializd!")
        
        for epoch in range(self.args.epochs):
            self.logger_dict["ep"] = epoch+1
            self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']
            
            # training and validation
            self.train_epoch()
            self.valid_epoch()
            self.lr_scheduler.step()

            # online log
            if self.args.wandb:
                self.wandb_logger.log(self.logger_dict)
            
            # terminal log
            columns = list(self.logger_dict.keys())
            values = list(self.logger_dict.values())
            print_table(values, columns, epoch, self.logger)

            # record time
            e_time = time.time() - start_time
            epoch_time.update(e_time)
            start_time = time.time()

            need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (self.args.epochs - epoch))
            print('[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs))

    
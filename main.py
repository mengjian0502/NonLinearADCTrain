"""
GPS-NN: Hardware-aware (channel) gating, (weight) pruning, and (gradient) skipping
"""

import os
import logging
import argparse
import torch
import models
from utils.loaders import get_loaders
from utils.utils import str2bool
from methods import BaseTrainer


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# loss and gradient
parser.add_argument('--loss_type', type=str, default='mse', help='loss func')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./dataset/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# acceleration
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# online logging
parser.add_argument("--wandb", type=str2bool, nargs='?', const=True, default=False, help="enable the wandb cloud logger")
parser.add_argument("--name")
parser.add_argument("--project")
parser.add_argument("--entity", default=None, type=str)

# gating
parser.add_argument('--cg_groups', type=int, default=1, help='apply channel gating if cg_groups > 1')
parser.add_argument('--cg_alpha', type=float, default=2.0, help='alpha value for channel gating')
parser.add_argument('--cg_threshold_init', type=float, default=0.0, help='initial threshold value for channel gating')
parser.add_argument('--target_cg_threshold', type=float, default=0.0, help='initial threshold value for channel gating')
parser.add_argument('--glambda', type=float, default=0, help='lambda for Channel Gating regularization')

# group lasso
parser.add_argument('--plambda', type=float, default=0, help='lambda for group lasso')
parser.add_argument('--wthre', type=float, default=0.0001, help='threshold of zeroing-out the groups')
parser.add_argument('--blk', type=int, default=1, help='weight block size')

# gradient skipping
parser.add_argument('--gthre', type=float, default=0.9, help='confidence threshold for gradient skipping')

# mixed precision training
parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable the mixed precision training or not")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()


def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    # initialize terminal logger
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    # get dataset
    num_classes, trainloader, testloader = get_loaders(args)

    # construct model
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"num_classes": num_classes, "args":args})
    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 
    logger.info(model)

    # initialize the trainer
    trainer = BaseTrainer(
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger
    )

    # start training
    trainer.fit()

if __name__ == '__main__':
    main()
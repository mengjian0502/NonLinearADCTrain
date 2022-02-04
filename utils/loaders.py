"""
Get dataloaders
"""
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def get_loaders(args):
    if args.dataset == 'cifar10':
        data_path = os.path.join(args.data_path, 'cifar')

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 10
    
    return num_classes, trainloader, testloader
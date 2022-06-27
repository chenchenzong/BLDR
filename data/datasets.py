import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
from data.augmentations import Augmentation, CutoutDefault
from data.augmentation_archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10


def input_dataset(dataset, noise_type, noise_path, is_human, cutout=0):
    if dataset == 'cifar10':
        train_cifar10_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_cifar10_aug = transforms.Compose([
            Augmentation(autoaug_paper_cifar10()),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if cutout > 0:
            train_cifar10_aug.transforms.append(CutoutDefault(cutout))

        train_dataset = CIFAR10('./dataset/', 
                                noise_type=noise_type, noise_path=noise_path, train=True, 
                                transform=train_cifar10_transform, 
                                transform_aug=train_cifar10_aug,
                                download=True
                               )
        test_dataset = CIFAR10('./dataset/', 
                               noise_type=noise_type, noise_path=noise_path, train=False, 
                               transform=test_cifar10_transform)
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_cifar100_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_cifar100_aug = transforms.Compose([
            Augmentation(autoaug_paper_cifar10()),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        if cutout > 0:
            train_cifar100_aug.transforms.append(CutoutDefault(cutout))

        train_dataset = CIFAR100('./dataset/', 
                                noise_type=noise_type, noise_path=noise_path, train=True, 
                                transform=train_cifar100_transform, 
                                transform_aug=train_cifar100_aug,
                                download=True
                               )
        test_dataset = CIFAR100('./dataset/', 
                               noise_type=noise_type, noise_path=noise_path, train=False, 
                               transform=test_cifar100_transform)
        num_classes = 100
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples









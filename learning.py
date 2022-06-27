# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models.resnet_for_selfKD import *
from models.resnet import *
from utils import *
import argparse
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--val_ratio', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)  # we will test your code with 5 different seeds. The seeds are generated randomly and fixed for all participants.
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0
    model.train()
    for i, (images, labels, indexes) in enumerate(train_loader):

        batch_size = indexes.shape[0]
       
        images =images.to(args.device)
        labels =labels.to(args.device)
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(loader, model, save=False):
    model.eval()    # Change model to 'eval' mode.
    
    pl_correct = 0
    nl_correct = 0
    correct = 0
    total = 0
    for images, _, labels, indexes, _ in loader:
        images = Variable(images).to(args.device)
        logits1, logits2, _, _, _, _, _, _, _ = model(images)
        outputs1 = F.softmax(logits1, dim=1)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        _, pred2 = torch.max(outputs2.data, 1)
        _, pred3 = torch.max((outputs1+outputs2).data, 1)
        total += labels.size(0)
        pl_correct += (pred1.cpu() == labels).sum()
        nl_correct += (pred2.cpu() == labels).sum()
        correct += (pred3.cpu() == labels).sum()
    pl_acc = 100*float(pl_correct)/float(total)
    nl_acc = 100*float(nl_correct)/float(total)
    acc = 100*float(correct)/float(total)

    return pl_acc


##################################### main code ################################################
args = parser.parse_args()
# Seed
set_global_seeds(args.seed)
args.device = set_device()
time_start = time.time()
# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


_, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset, args.noise_type, args.noise_path, True)
# load model
print('building model...')
model = multi_resnet34_kd(num_classes)
print('building model done')



test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)




# we will test the model by the following code
state_dict = torch.load(f"ckpts/{args.dataset}_{args.noise_type}best.pth.tar", map_location = "cpu")
model.load_state_dict(state_dict['state_dict'])
model.to(args.device)
test_acc = evaluate(loader=test_loader, model=model, save = False)
print(f'Best test acc selected by val is {test_acc}')

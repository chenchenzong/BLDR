# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models.resnet_for_selfKD import *
from models.resnet import *
from loss import *
from utils import *
import argparse
import time
import os

import numpy as np
import copy



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.02)
parser.add_argument('--val_ratio', type = float, default = 0.05)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--seed', type=int, default=0)  # we will test your code with 5 different seeds. The seeds are generated randomly and fixed for all participants.
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--batchsize', type = int, default=256)
parser.add_argument('--is_human', action='store_true', default=True)
parser.add_argument('--ckpt_num', type = int, default=3)

## reparam_arch
parser.add_argument('--mean', type = float, default = 0.0)
parser.add_argument('--std', type = float, default = 1e-8)
parser.add_argument('--lr_u', type = float, default = 10)
parser.add_argument('--lr_v', type = float, default = 10)
parser.add_argument('--ratio_consistency', type = float, default = 0.9)
parser.add_argument('--ratio_balance', type = float, default = 0.1)
parser.add_argument('--ratio_reg', type = float, default = 25)

## data_augmentation
parser.add_argument('--aug_type', type = str, default="autoaug_cifar10")
parser.add_argument('--cutout', type = int, default=16)
parser.add_argument("--updateW_epochs", type=int, default=20)

#kd parameter
parser.add_argument('--temperature', default=3, type=int, help='temperature to smooth the logits')
parser.add_argument('--alpha', default=0.1, type=float, help='weight of kd loss')
parser.add_argument('--beta', default=1e-6, type=float, help='weight of feature loss')

# Train the Model
def train(args, epoch, num_classes, train_loader, model, train_criterion, optimizer, optimizer_u, optimizer_v, threshold=0.3):
    updateW_epochs = args.updateW_epochs
    ratio_consistency = args.ratio_consistency
    
    model.train()
            
    train_total=0
    pl_train_correct=0
    nl_train_correct=0

    nl_probs = []
    nl_idxs = []

    for i, (data, data2, label, indexes, _) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
       
        data, label, indexes = data.to(args.device), label.long().to(args.device), indexes.to(args.device)

        target = torch.zeros(len(label), num_classes).to(args.device).scatter_(1, label.view(-1,1), 1)
        targets_neg = torch.ones(len(label), num_classes).to(args.device).scatter_(1, label.view(-1,1), 0)
        
        if ratio_consistency > 0:
            data2 = data2.to(args.device)
            data_all = torch.cat([data, data2]).to(args.device)
        else:
            data_all = data
                    
        output1, output2, middle_output1, middle_output2, middle_output3, _, _, _, _ = model(data_all)
        logits1, logits2, middle_logits1, middle_logits2, middle_logits3, final_fea, middle1_fea, middle2_fea, middle3_fea = model(data)
        _, pred = logits2.max(1)
        targets_neg.scatter_(1, pred.view(-1,1), 0)
        loss_pl = train_criterion(indexes, output1, None, target, None, True)
        loss_nl = train_criterion(indexes, None, output2, target, targets_neg)

        middle1_loss = train_criterion(indexes, middle_output1, None, target, None, True)
        middle2_loss = train_criterion(indexes, middle_output2, None, target, None, True)
        middle3_loss = train_criterion(indexes, middle_output3, None, target, None, True)

        temp4 = ((logits1 + logits2)/2) / args.temperature
        temp4 = torch.softmax(temp4, dim=1)
        loss1by4 = kd_loss_function(middle_logits1, temp4.detach(), args) * (args.temperature**2)
        loss2by4 = kd_loss_function(middle_logits2, temp4.detach(), args) * (args.temperature**2)
        loss3by4 = kd_loss_function(middle_logits3, temp4.detach(), args) * (args.temperature**2)
        feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
        feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
        feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 
        
        total_loss = loss_nl + ((loss_pl + middle1_loss + middle2_loss + middle3_loss) + \
                            args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                            args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3))
        if num_classes == 100:
            total_loss += 10*nn.KLDivLoss()(F.log_softmax(logits2), F.softmax(logits1))
        
        optimizer_u.zero_grad()
        optimizer_v.zero_grad()
        optimizer.zero_grad()
        
        total_loss.backward()

        optimizer_u.step()
        optimizer_v.step()
        optimizer.step()
        
        nl_probs.extend(F.softmax(logits2, dim=1)[range(len(label)), label].cpu().detach().numpy().tolist())
        nl_idxs.extend(indexes.cpu().numpy().tolist())
        # Forward + Backward + Optimize
        prec1, _ = accuracy(logits1, label, topk=(1, 5))
        prec2, _ = accuracy(logits2, label, topk=(1, 5))

        # prec = 0.0
        train_total+=1
        pl_train_correct+=prec1
        nl_train_correct+=prec2
        
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] PL Training Accuracy: %.4F, NL Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch + updateW_epochs, i+1, len(train_dataset)//batch_size, prec1, prec2, total_loss.data))

    pl_train_acc=float(pl_train_correct)/float(train_total)
    nl_train_acc=float(nl_train_correct)/float(train_total)

    nl_probs = np.asarray(nl_probs)[np.argsort(nl_idxs)]
    if epoch >= updateW_epochs:
        nl_max = nl_probs.max()
        nl_min = nl_probs.min()
        weight = (nl_probs - nl_min)/(nl_max - nl_min)
        train_criterion.update_weight(weight)
        train_criterion.ratio = len(nl_probs[nl_probs<threshold])/len(nl_idxs)
        print(len(nl_probs[nl_probs<threshold])/len(nl_idxs))
   
    return pl_train_acc, nl_train_acc

# Evaluate the Model
def evaluate(loader, model):
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

    return pl_acc, nl_acc, acc

def Recognize(args, data_loader, model, threshold=0.05):
    model.eval()    # Change model to 'eval' mode.

    probs = []
    idxs = []
    with torch.no_grad():
        for images, _, labels, indexes, _ in data_loader:
            images = Variable(images).to(args.device)
            _, logits2, _, _, _, _, _, _, _ = model(images)
            
            prob = F.softmax(logits2, dim=1)
            probs.extend(prob.cpu().detach().numpy().tolist())
            idxs.extend(indexes.numpy().tolist())

    global noise_or_not, noisy_targets
    tmp_idxs = np.asarray(idxs)
    tmp_probs = np.asarray(probs)
    idxs = tmp_idxs[np.argsort(tmp_idxs)]
    probs = tmp_probs[np.argsort(tmp_idxs)]
    preds = []
    for i in range(len(idxs)):
        preds.append(probs[i][noisy_targets[i]])
    preds = np.asarray(preds)

    noise_idxs = np.arange(len(noise_or_not))[noise_or_not==1]

    noisy_or_not_predict = copy.deepcopy(noise_or_not)
    noisy_or_not_predict[preds<threshold] = True
    noisy_or_not_predict[preds>=threshold] = False
    np.save("detection.npy", noisy_or_not_predict)

##################################### main code ################################################
args = parser.parse_args()

# Seed
set_global_seeds(args.seed)
args.device = set_device()
time_start = time.time()
# Hyper Parameters
batch_size = args.batchsize
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


train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset, args.noise_type, args.noise_path, args.is_human, args.cutout)
# print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model

noise_or_not = train_dataset.noise_or_not
noisy_targets = train_dataset.train_noisy_labels

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = args.batchsize,
                                   num_workers=args.num_workers,
                                   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 200,
                                  num_workers=args.num_workers,
                                  shuffle=False)  

print('building model...')
model = multi_resnet34_kd(num_classes)
model = nn.DataParallel(model)
print('building model done')
model.to(args.device)
ema = EMA(model)

train_loss = BLDR_loss(num_training_samples, num_classes, args.mean, args.std, args.ratio_consistency, args.ratio_balance, args.ratio_reg).to(args.device)
                                                                            
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
optimizer_u = torch.optim.SGD([train_loss.u1], lr=args.lr_u, weight_decay=0, momentum=0)
optimizer_v = torch.optim.SGD([train_loss.v1], lr=args.lr_v, weight_decay=0, momentum=0)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.0002)

best_acc = 0
for epoch in range(args.n_epoch+args.updateW_epochs):
# train models
    print(f'epoch {epoch}')
    pl_train_acc, nl_train_acc = train(args, epoch, num_classes, train_loader, model, train_loss, optimizer, optimizer_u, optimizer_v)
    if epoch >= args.updateW_epochs:
        lr_scheduler.step()
    # evaluate models
    pl_test_acc, nl_test_acc, test_acc = evaluate(test_loader, model)

    # EMA
    ema.decay = min(0.9,(1+epoch)/(10+epoch))
    if epoch == 0:
        ema.register()
    else:
        ema.update()
    ema.apply_shadow()
    state = {'state_dict': model.state_dict(),
                        'epoch':epoch,
                        'acc':0.0,
                }
    save_path= os.path.join('ckpts/', args.dataset + '_' + args.noise_type +'best.pth.tar')
    torch.save(state,save_path)
    ema_pl_test_acc, ema_nl_test_acc, ema_test_acc = evaluate(test_loader, model)
    
    ema.restore()
    # save results
    print("Learning rate: {}.".format(optimizer.state_dict()['param_groups'][0]['lr']))
    print('PL train acc: {}, NL train acc: {}.'.format(round(pl_train_acc,2), round(nl_train_acc,2)))
    print('PL test acc: {}, NL test acc: {}, Ensemble test acc:{}.'.format(round(pl_test_acc,2), round(nl_test_acc,2),round(test_acc,2)))
    print('EMA PL test acc: {}, EMA NL test acc: {}, EMA Ensemble test acc:{}.'.format(round(ema_pl_test_acc,2), round(ema_nl_test_acc,2),round(ema_test_acc,2)))
    
    time_curr = time.time()
    time_elapsed = time_curr - time_start
    print(f'[Epoch {epoch}] Time elapsed {time_elapsed//3600:.0f}h {(time_elapsed%3600)//60:.0f}m {(time_elapsed%3600)%60:.0f}s', flush=True)

Recognize(args, train_loader, model)
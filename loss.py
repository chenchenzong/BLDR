import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions


def loss_cross_entropy(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    ## Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch


def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


class BLDR_loss(nn.Module):
    def __init__(self, num_examp, num_classes, mean, std, ratio_consistency = 0, ratio_balance = 0, ratio_reg = 100):
        super(BLDR_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance
        self.ratio_reg = ratio_reg

        self.u1 = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v1 = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=mean, std=std)
        
        self.weight = torch.tensor([1.0 for i in range(num_examp)])
        self.ratio = 0.1

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u1, mean=mean, std=std)
        torch.nn.init.normal_(self.v1, mean=mean, std=std)

    def update_weight(self, weight):
        self.weight = torch.tensor(weight)

    def forward(self, index, outputs, nl_outputs, label, labels_neg, flag=False):
        if flag:
            return self.forward_wo_nl(index, outputs, label)
        else:
            return self.forward_nl(index, nl_outputs, label, labels_neg)

    def consistency_loss(self, index, output1, output2):            
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1,1), 1)

    def forward_wo_nl(self, index, outputs, label):
            
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            ensembled_output = 0.5 * (output + output2).detach()

        else:
            output = outputs
            ensembled_output = output.detach()

        U_square1 = self.u1[index]**2 * label 
        V_square1 = self.v1[index]**2 * (1 - label) 
        U_square1 = torch.clamp(U_square1, 0, 1-1e-4)
        V_square1 = torch.clamp(V_square1, 0, 1-1e-4)
        E1 =  U_square1 - V_square1
        self.E1 = E1

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(original_prediction + U_square1 - V_square1.detach(), min = 1e-4)
        prediction = F.normalize(prediction, p = 1, eps = 1e-4)
        prediction = torch.clamp(prediction, min = 1e-4, max = 1.0)
        label_one_hot = self.soft_to_hard(output.detach())

        if self.ratio_reg > 0:
            pl_loss = F.mse_loss((label_one_hot + U_square1.detach() - V_square1), label,  reduction='sum') / len(label) 
            MSE_loss = pl_loss

        else:
            MSE_loss = torch.zeros(1)
        
        loss_pos = torch.mean(-torch.sum(self.weight[index].cuda().reshape(-1,1)*(label)*torch.log(prediction), dim = -1))
        loss = loss_pos

        if self.ratio_reg > 0:

            loss += self.ratio_reg * self.ratio * self.ratio * MSE_loss 


        if self.ratio_balance > 0:
            pl_avg_prediction = torch.mean(prediction, dim=0)
            pl_prior_distr = 1.0/self.num_classes * torch.ones_like(pl_avg_prediction)
            pl_avg_prediction = torch.clamp(pl_avg_prediction, min = 1e-4, max = 1.0)
            pl_balance_kl =  torch.mean(-(pl_prior_distr * torch.log(pl_avg_prediction)).sum(dim=0))
            loss += self.ratio_balance * (pl_balance_kl)

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):

            pl_consistency_loss = self.consistency_loss(index, output, output2)

            loss += self.ratio_consistency * (torch.mean(pl_consistency_loss))

        return  loss


    def forward_nl(self, index, nl_outputs, label, labels_neg):
        if len(nl_outputs) > len(index):
            nl_output, nl_output2 = torch.chunk(nl_outputs, 2)

        else:
            nl_output = nl_outputs



        nl_original_prediction = F.softmax(nl_output, dim=1)
        nl_label_one_hot = self.soft_to_hard(nl_output.detach())

        s_neg = torch.log(torch.clamp(1.-nl_original_prediction, min=1e-5, max=1.))

        loss_neg = torch.mean(-torch.sum(self.weight[index].cuda().reshape(-1,1)*(labels_neg)*s_neg, dim = -1))

        loss = loss_neg


        if self.ratio_balance > 0:
            nl_avg_prediction = torch.mean(nl_original_prediction, dim=0)
            nl_prior_distr = 1.0/self.num_classes * torch.ones_like(nl_avg_prediction)
            nl_avg_prediction = torch.clamp(nl_avg_prediction, min = 1e-4, max = 1.0)
            nl_balance_kl =  torch.mean(-(nl_prior_distr * torch.log(nl_avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * (nl_balance_kl)

        if (len(nl_outputs) > len(index)) and (self.ratio_consistency > 0):

            nl_consistency_loss = self.consistency_loss(index, nl_output, nl_output2)

            loss += self.ratio_consistency * (torch.mean(nl_consistency_loss))

        return  loss

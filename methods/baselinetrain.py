import backbone
import utils
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, res_model, loss_type = 'softmax', feature_aug=False, ratio=0, radius=10):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func(avg_pool=True)
        self.feature_aug = feature_aug
        self.ratio = ratio
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            #self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
            self.classifier = backbone.distLinear(640, num_class)
        elif loss_type == 'norm':
            self.classifier = backbone.NormLinear(self.feature.final_feat_dim, num_class, radius=radius)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.rank = rank
        self.world_size = world_size
        self.DBval = False; #only set True for CUB dataset, see issue #31

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        return out



    def forward_loss(self, feature, y):
        scores  = self.classifier.forward(feature)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y )
                
    def train_loop(self, epoch, train_loader, optimizer, tb_logger):
        print_freq = 10
        ori_avg_loss=0
        aug_avg_loss=0
        
        ratio = self.ratio

        for i, (x, y) in enumerate(train_loader):
            ori_feature = self.forward(x)
            ori_loss  = self.forward_loss(ori_feature, y)
            if self.feature_aug:
                aug_feature = ori_feature + torch.randn_like(ori_feature) * 0.5
                aug_loss  = self.forward_loss(aug_feature, y)
                loss = ori_loss + aug_loss
            else:
                aug_loss = ori_loss
                loss = ori_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ori_avg_loss = ori_avg_loss+ori_loss.item()
            aug_avg_loss = aug_avg_loss+aug_loss.item()

            bs = x.size(0)

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Ori Loss {:f} | Aug Loss {:f}'.format(epoch, i, len(train_loader), ori_avg_loss/float(i+1), aug_avg_loss/float(i+1)))
                curr_step = epoch*len(train_loader) + i
                tb_logger.add_scalar('Ori Loss', ori_avg_loss/float(i+1), curr_step)
                     


    def analysis_loop(self, val_loader, record = None):
        cls_class_file  = {}
        #classifier = self.classifier.weight.data
        for i, (x,y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.feature.forward(x_var)

            cls_feats = feats.data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(cls_feats, labels):
                if l not in cls_class_file.keys():
                    cls_class_file[l] = []
                cls_class_file[l].append(f)
        for cl in cls_class_file:
            cls_class_file[cl] = np.array(cls_class_file[cl])

        DB, intra_dist, inter_dist = DBindex(cls_class_file)
        #sum_dist = get_dist(classifier)
        print('DB index (cls) = %4.2f, intra_dist (cls) = %4.2f, inter_dist (cls) = %4.2f' %(DB, intra_dist, inter_dist))
        return 1/DB #DB index: the lower the better






def DBindex(cl_data_file):
    #For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    #DB index present the intra-class variation of the data
    #As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    #Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    intra_dist = []
    inter_dist = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
        intra_dist.append(stds[i])
        inter_dist.append(np.mean([mdists[i,j] for j in range(cl_num) if j != i]))

    return np.mean(DBs), np.mean(intra_dist), np.mean(mdists)

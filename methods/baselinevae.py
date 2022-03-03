import backbone
import utils
import pdb
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional
from utils import l2_norm, _get_log_pz_qz_prodzi_qzCx
import torch.distributed as dist

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x




class DisentangleNet(nn.Module):
    def __init__(self, model_func,  num_class, kl_weight=1, loss_type = 'softmax', aug_weight=0, use_conv=False, rank=0, world_size=0, avg=True):
        super(DisentangleNet, self).__init__()
        self.backbone = model_func(flatten=False)
        self.aug_weight = aug_weight
        self.DBval = True
        self.rank = rank
        self.world_size = world_size
        self.use_conv = use_conv
        if not use_conv:
            channel = 640
            pool_size = 5
            feature_dim = 640
            self.cls_fc = nn.Sequential(
                nn.AvgPool2d(pool_size),
                backbone.Flatten()      
            )
        else:
            channel = 64
            pool_size = 5
            feature_dim = 1600
            self.cls_fc = nn.Sequential(
                backbone.Flatten()      
            )

        cls_feature_dim = feature_dim
        self.encoder =  nn.Sequential(
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            backbone.Flatten(),
        )
        self.vae_mean = nn.Linear(channel*pool_size*pool_size, feature_dim)
        self.vae_var = nn.Linear(channel*pool_size*pool_size, feature_dim)
        self.decoder_fc = nn.Linear(feature_dim, channel*pool_size*pool_size)
        self.decoder =  nn.Sequential(
                Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
                Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
                Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        )
        self.superloss = torch.nn.MSELoss()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(cls_feature_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(cls_feature_dim, num_class)
        else:
            self.classifier = backbone.NormLinear(cls_feature_dim, num_class, radius=10)
        self.smloss = torch.nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)
        # remove abnormal points
        return mu + eps*std
    
    def forward_d(self, aggr_feature):
        if not self.use_conv:
            channel = 640
            pool_size = 5
        else:
            channel = 64
            pool_size = 5
        recon_feature = self.decoder_fc(aggr_feature).view(-1, channel, pool_size, pool_size)
        recon_feature = self.decoder(recon_feature)
        return recon_feature

    def forward(self, x): 
        feature_map = self.backbone(x)
        cls_feature = self.cls_fc(feature_map)
        #if self.use_wide:
        #    feature_map = self.downsample(feature_map)
        bs = x.size(0)
        if not self.use_conv:
          channel = 640
        else:
          channel = 64
        encoder_map = feature_map + cls_feature.view(bs, channel, 1, 1).detach()
        #encoder_map = feature_map
        encoder_feature = self.encoder(encoder_map)
        mu = self.vae_mean(encoder_feature)
        logvar = self.vae_var(encoder_feature)
        return cls_feature, mu, logvar, feature_map
        #return cls_feature


    def train_all(self, epoch, train_loader, optimizer, tb_logger, n_data=None):
        print_freq = 10
        cls_avg_loss = 0   
        recon_avg_loss = 0
        kl_avg_loss = 0
        aug_avg_loss = 0
        beta = self.kl_weight 

        for i, (x, y) in enumerate(train_loader):
            x = Variable(x.cuda())
            bs = x.size(0)
            cls_feature, mu, logvar, feature_map = self.forward(x)
            scores = self.classifier.forward(cls_feature)
            y = Variable(y.cuda())           

            # classification loss
            cls_loss = self.smloss(scores, y)
            cls_invar_feature = self.reparameterize(mu, logvar)
            aggr_feature = cls_feature + cls_invar_feature.detach()

            # reconstruction loss
            recon_feature = self.forward_d(aggr_feature)
            recon_loss = self.superloss(recon_feature, feature_map.detach())
            #recon_loss = self.superloss(recon_feature, x)
            # feature aug loss
            if self.aug_weight > 0:
                aug_cls_feature = cls_feature + cls_invar_feature
                aug_scores  = self.classifier.forward(aug_cls_feature)
                aug_cls_loss = self.smloss(aug_scores, y)
            else:
                aug_cls_loss = torch.zeros(1).cuda()

            # kl_loss
            #kl_loss = -0.5*torch.sum(1+logvar-logvar.exp()-mu.pow(2)) / bs
            log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(cls_invar_feature, (mu, logvar), n_data, is_mss=False)
            #I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
            mi_loss = (log_q_zCx - log_qz).mean()
            # TC[z] = KL[q(z)||\prod_i z_i]
            tc_loss = (log_qz - log_prod_qzi).mean()
            # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
            dw_kl_loss = (log_prod_qzi - log_pz).mean()
            #loss = cls_loss + recon_loss + kl_loss * self.kl_weight + aug_cls_loss * self.aug_weight
            loss = cls_loss + recon_loss + mi_loss  + dw_kl_loss + tc_loss * beta  + aug_cls_loss * self.aug_weight
            #loss = cls_loss + recon_loss + kl_loss * self.kl_weight
            kl_loss = mi_loss + dw_kl_loss + tc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == y).float())

            cls_avg_loss = cls_avg_loss+cls_loss.item()
            recon_avg_loss = recon_avg_loss+recon_loss.item()
            kl_avg_loss = kl_avg_loss+kl_loss.item()        
            aug_avg_loss = aug_avg_loss+aug_cls_loss.item()

            if i%print_freq==0 and self.rank == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Cl Loss {:f} | Recon Loss {:f} | Kl Loss {:f} | Aug Loss {:f}'.format(epoch, i, len(train_loader), cls_avg_loss/float(i+1), recon_avg_loss/float(i+1), kl_avg_loss/float(i+1), aug_avg_loss/float(i+1)))
                curr_step = epoch*len(train_loader) + i
                tb_logger.add_scalar('Cl Loss', cls_avg_loss/float(i+1), curr_step)
                tb_logger.add_scalar('Recon Loss', recon_avg_loss/float(i+1), curr_step)
                tb_logger.add_scalar('KL Loss', kl_avg_loss/float(i+1), curr_step)
                tb_logger.add_scalar('Aug Loss', aug_avg_loss/float(i+1), curr_step)

  


    def analysis_loop(self, val_loader, record = None):
        cls_class_file  = {}
        #classifier = self.classifier.weight.data
        for i, (x,y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.backbone.forward(x_var)
            feats = self.cls_fc(feats)

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


                            

def DBindex(cls_data_file):
    #For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    #DB index present the intra-class variation of the data
    #As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    #Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cls_data_file.keys()
    cls_num= len(class_list)
    cls_means = []
    stds = []
    DBs = []
    intra_dist = []
    inter_dist = []
    for cl in class_list:
        cls_means.append( np.mean(cls_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cls_data_file[cl] - cls_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cls_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cls_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cls_num) if j != i ]) )
        intra_dist.append(stds[i])
        inter_dist.append(np.mean([mdists[i,j] for j in range(cls_num) if j != i]))

    return np.mean(DBs), np.mean(intra_dist), np.mean(mdists)

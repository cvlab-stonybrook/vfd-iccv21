import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py
import pdb
import random
import time
import configs
import backbone
from data.datamgr import SimpleDataManager
from data import feature_loader
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.baselinevae import DisentangleNet
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
from utils import l2_norm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import entropy
from scipy.special import softmax

def aug_features(all_cls_feature, mu, logvar, feature_map, model, aug_per_sample=2, n_way=5, n_shot=1, feat_dim=512):
    all_cls_feature = all_cls_feature.view(n_way, n_shot, feat_dim)
    mu = mu.view(n_way, n_shot, feat_dim)
    logvar = logvar.view(n_way, n_shot, feat_dim)
    aug_features = torch.zeros((n_way, aug_per_sample*n_shot, feat_dim))
    aug_y= torch.from_numpy(np.repeat(range( n_way ), aug_per_sample*n_shot))
    aug_y = Variable(aug_y).cuda()
    for cls in range(n_way):  
        cls_feature = all_cls_feature[cls,:,:]
        cls_feature = cls_feature.repeat(aug_per_sample, 1)
        cls_mu = mu[cls,:,:]
        #cls_mu = cls_mu.mean(0, True)
        cls_mu = cls_mu.repeat(aug_per_sample, 1)
        cls_logvar = logvar[cls,:,:]
        #cls_logvar = cls_logvar.mean(0, True)
        cls_logvar = cls_logvar.repeat(aug_per_sample, 1)
        cls_invar_feature = torch.randn_like(cls_feature) 
        #cls_invar_feature = model.reparameterize(cls_mu, cls_logvar)
        #cls_feature_map = feature_map[cls,:,:,:]
        #cls_feature_map = cls_feature_map.unsqueeze(0)
        #cls_invar_feature = torch.randn_like(cls_feature) * 0.2
        aggr_feature = cls_feature + cls_invar_feature  
        #recon_feature = model.forward_d(aggr_feature)
        #recon_loss = model.superloss(recon_feature, cls_feature_map)
        #cls_aug_feature = model.cls_fc(recon_feature)
        cls_aug_feature = aggr_feature
        aug_features[cls, :, :] = cls_aug_feature
    aug_features = aug_features.view(n_way*aug_per_sample*n_shot, -1)
    return aug_features.detach().cuda(), aug_y




def finetune_backbone(model, img_data_file, n_way, n_support, feat_dim, aug=False, n_query=15):

    class_list = img_data_file.keys()
    select_class = random.sample(class_list, n_way)
    img_all = []

    for cl in select_class:
        img_data = img_data_file[cl]
        img_data = np.array(img_data)
        perm_ids = np.random.permutation(len(img_data)).tolist()
        perm_ids = perm_ids + perm_ids
        img_all.append( [ np.squeeze( img_data[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch
    
    # samples images for support set and query set
    img_all = torch.from_numpy(np.array(img_all))
    img_all = Variable(img_all).cuda()
    [c, h, w] = img_all[0][0].shape

    x_support = img_all[:, :n_support,:,:,:]
    x_query = img_all[:,n_support:,:,:,:]

    x_support   = x_support.contiguous().view(n_way* n_support, c, h, w)
    x_query     = x_query.contiguous().view(n_way* n_query, c, h, w)
   
    y_support = torch.from_numpy(np.repeat(range( n_way ), n_support ))
    y_support = Variable(y_support).cuda()
    
    z_support, mu, logvar, feature_map = model(x_support) 
    z_query, _, _, _ = model(x_query) 
    #z_support = model(x_support).detach().cuda()
    z_support = z_support.view(n_way*n_support, -1).detach().cuda()
    z_query = z_query.view(n_way*n_query, -1).detach().cuda()
    #z_mean = torch.mean(z_support, dim=1)
    #_, z_mean = l2_norm(z_mean)

    #z_query = model(x_query).detach().cuda()
      #z_query = z_query.view(n_way, n_query, -1)

    #aug_z = recon_z.view(n_way*n_support, -1).detach().cuda()
    #aug_y = y_support.clone()
    #z_support_all = z_support.view(n_way*n_support, -1)
    feat_dim = 640
    y_query = np.repeat(range( n_way ), n_query )
    aug_z, aug_y = aug_features(z_support, mu, logvar, feature_map, model, n_shot=n_support, aug_per_sample=params.aug_per_sample, feat_dim=feat_dim)
    #aug_z, aug_y = trans_features(aug_z, aug_y, z_query, z_support)
    #cls_invar = model.reparameterize(mu, logvar)
    #aug_z, aug_y = aug_beta_features(z_support, cls_invar, feature_map, model, n_shot=n_support)
    if aug_z is not None:
      z_support_all = torch.cat((z_support, aug_z))
      y_support_all = torch.cat((y_support, aug_y))
    else:
      z_support_all = z_support
      y_support_all = y_support
    #z_support_all = z_support
    #y_support_all = y_support
    # train classifier with augmened features
    linear_clf = backbone.distLinear(feat_dim, n_way).cuda()
    ## initialize weights for linear_clf
    set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    #    
    batch_size = 4
    support_size = z_support_all.shape[0]
    for epoch in range(100):
        rand_id = np.random.permutation(support_size)
        for i in range(0, support_size , batch_size):
           set_optimizer.zero_grad()
           selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
           z_batch = z_support_all[selected_id]
           y_batch = y_support_all[selected_id] 
           scores  = linear_clf(z_batch)
           loss = loss_function(scores,y_batch)
           loss.backward()
           set_optimizer.step()
    #    
    model.eval()
    scores = linear_clf(z_query)
    pred = torch.argmax(scores, 1)
    #acc = np.mean(np.array(pred.cpu().data) == y_query)
    scores = linear_clf(z_query)
    pred = torch.argmax(scores, 1)
    #pdb.set_trace()
    acc = np.mean(np.array(pred.cpu().data) == y_query)*100
    return acc
    

if __name__ == '__main__':
    params = parse_args('finetune_sample')

    image_size = 84


    split = "novel"
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(params.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += "_" + params.split
    checkpoint_dir += '_%.2f'%(params.kl_weight)
    if params.assign_name is not None:
        modelfile   = get_assigned_file(checkpoint_dir,params.assign_name)
#    elif params.method in ['baseline', 'baseline++'] :
#        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
    else:
        modelfile   = get_best_file(checkpoint_dir)
    if params.save_iter != -1:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 

    datamgr         = SimpleDataManager(image_size, batch_size = 64)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False, shuffle=False)
    img_data_file = feature_loader.init_img_loader(data_loader)
    
    #base_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), "base.hdf5") #defaut split = novel, but you can also test base or val classes
    #base_data_file = feature_loader.init_loader(base_file)

    model           = DisentangleNet( model_dict[params.model], params.num_classes, kl_weight=params.kl_weight, loss_type = params.loss_type)

    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "module." in key:
            newkey = key.replace("module.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
    #    else:
    #        state.pop(key)

    model = model.cuda()        
     
    dirname = os.path.dirname(novel_file)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    iter_num = 600
    acc_all = []
    model.load_state_dict(state, strict=False)
    #visualize_intra_cls_var(model)
    for i in range(iter_num):
        acc = finetune_backbone(model, img_data_file, params.train_n_way, params.n_shot, 512)
        acc_all.append(acc)
        if i%10 == 0:
            print('Iter: %d, Acc : %f, Avg Acc: %f'% (i, acc, np.mean(np.array(acc_all))))
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)

    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))


    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        if False :
            exp_setting = '%s-%s-%s-%s %sshot %sway_test' %(params.dataset,  params.model, params.method, aug_str, params.n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s%s %sshot %sway_train %sway_test aug%s' %(params.dataset, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way, str(params.aug_per_sample) )
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import time
import os
import glob
import pdb

import configs
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.baselinevae import DisentangleNet

from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file 

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, tb_logger):    

    cls_params_ids = []
    params_ids1 = [id(p) for p in model.backbone.parameters()]
    cls_params_ids.extend(params_ids1)
    params_ids2 = [id(p) for p in model.classifier.parameters()]

    cls_params_ids.extend(params_ids2)
             

    cls_params = [p for p in model.parameters() if id(p) in cls_params_ids and p.requires_grad]
    g_params = [p for p in model.parameters() if id(p) not in cls_params_ids and p.requires_grad]
    optimizer = torch.optim.Adam([
           {'params': cls_params, 'lr': 0.001},
           {'params': g_params, 'lr': 0.0001}
    ])
    max_acc = 0       
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        if params.lr_steps is not None and epoch in params.lr_steps:
            for param_group in optimizer.param_groups:
                init_lr = param_group['lr']
                param_group['lr'] = init_lr * 0.1
        
        model.train_all(epoch, base_loader, optimizer, tb_logger, len(base_loader)*params.bs)

        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        acc = model.analysis_loop(val_loader)

        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if ((epoch % params.save_freq==0) or (epoch==stop_epoch-1)):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':

    np.random.seed(10)
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + params.split + '.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 

    if 'Conv' in params.model or 'ResNet12' in params.model:
        image_size = 84
    else:
        image_size = 224


    optimization = 'Adam'


    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.bs)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.bs)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        

        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
        else:
            model           = DisentangleNet( model_dict[params.model], params.num_classes, kl_weight=params.kl_weight, aug_weight=params.aug_weight, loss_type = params.loss_type)



    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(params.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    params.checkpoint_dir += '_' + params.split
    params.checkpoint_dir += '_%.2f'%(params.kl_weight)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    tb_logger = SummaryWriter('%s/events/%s' %(params.save_dir, time.time()))

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir, params.resume_iter)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            state = tmp['state']
            state_keys = list(state.keys())
            model.load_state_dict(state, strict=True)
            keys1 = set([k for k,_ in model.named_parameters()])
            keys2 = set(tmp['state'].keys())
            not_loaded = keys2 - keys1
            for k in not_loaded:
                print('caution: {} not loaded'.format(k))

    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        warmup_resume_file = get_assigned_file(params.checkpoint_dir, str(params.warmup_file))
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.backbone.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    if params.evaluate:        
        eval_file = os.path.join(params.checkpoint_dir, params.evaluate)
        if eval_file is not None:
            tmp = torch.load(eval_file)
            model.load_state_dict(tmp['state'])
        model.eval()  
        acc = model.get_cov(val_loader)
    else:
        print(params)
        model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params, tb_logger)

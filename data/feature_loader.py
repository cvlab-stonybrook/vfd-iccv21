import torch
import numpy as np
import h5py
import pdb
class SimpleHDF5Dataset:
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
           # print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_img_loader(data_loader):
      
    img_data_file = {}
    
    for i, (x, y) in enumerate(data_loader):
        bs = x.size(0)
        for idx in range(bs):
            y_idx = y[idx].item()
            x_idx = np.array(x[idx])
            if y_idx not in img_data_file.keys():  
                img_data_file[y_idx] = []
            img_data_file[y_idx].append(x_idx)
    
    return img_data_file

def init_loader(filename):
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    #labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    while np.sum(feats[-1]) == 0:
        feats  = np.delete(feats,-1,axis = 0)
        labels = np.delete(labels,-1,axis = 0)
        
    class_list = np.unique(np.array(labels)).tolist() 
    inds = range(len(labels))

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in inds:
        cl_data_file[labels[ind]].append( feats[ind])

    return cl_data_file


def get_classmap(dset):
    '''
    Creates a mapping between serial number of a class
    in provided dataset and the indices used for classification.
    Returns:
        2 dicts, 1 each for train and test classes
    '''
    class_names_file = '/home/jingyi/feature_generating_pytorch/datasets/%s/classes.txt' % dset
      
    with open(class_names_file) as fp:
        all_classes = fp.readlines()
    with open('/home/jingyi/feature_generating_pytorch/datasets/%s/testclasses.txt' % dset) as fp:
        test_class_names = [i.strip() for i in fp.readlines() if i != '']

    test_count = 0
    train_count = 0

    train_classmap = dict()
    test_classmap = dict()
    for line in all_classes:
        idx, name = [i.strip() for i in line.split(' ')]
        if name in test_class_names:
            test_classmap[int(idx)] = test_count
            test_count += 1
        else:
            train_classmap[int(idx)] = train_count
            train_count += 1
    return train_classmap, test_classmap

def load_feat(features, labels, train_classmap, test_classmap):
    cl_data_file = {}
    for feat, label in zip(features, labels):
        if label in test_classmap.keys():
          if label not in cl_data_file.keys():
            cl_data_file[label] = feat.reshape((1, -1))
          else:
            cl_data_file[label] = np.concatenate([cl_data_file[label], feat.reshape((1, -1))], 0)
    return cl_data_file

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.manifold import TSNE
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import pdb

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax"):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type

    def set_forward(self,x,is_feature = True, aug_per_sample=0):
        return self.set_forward_adaptation(x,is_feature, aug_per_sample=aug_per_sample); #Baseline always do adaptation
 
    def set_forward_adaptation(self,x,is_feature = True, base_cl_data_file=None, aug_per_sample=0):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)
        if aug_per_sample > 0:
          z_support = self.aug_features(z_support, aug_per_sample=aug_per_sample)
        z_support_all   = z_support.contiguous().view(self.n_way* (self.n_support+aug_per_sample), -1 )
        z_query_all     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        y_support_all = torch.from_numpy(np.repeat(range( self.n_way ), (self.n_support+aug_per_sample) ))
        y_support_all = Variable(y_support_all.cuda())
        y_query_all = np.repeat(range(self.n_way), self.n_query)
        batch_size = 4

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        #support_size = self.n_way* self.n_support
        support_size = z_support_all.shape[0]
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support_all[selected_id]
                y_batch = y_support_all[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
            #scores = linear_clf(z_support_all)
            #pred = torch.argmax(scores, dim=1)
            #acc = np.mean(np.array((pred == y_support_all).cpu().data))
            #print('Epoch: %d, Acc: %f'%(epoch, acc))
        scores = linear_clf(z_query_all)
        #pdb.set_trace()
        #weight_embedded = TSNE(n_components=2).fit_transform(linear_clf.L.weight.cpu().data)
        #plot_tsne(z_embedded, np.array(y_support_all.cpu().data), weight=weight_embedded)
        return scores

    def aug_features(self, ori_features, aug_per_sample, n_way=5, n_shot=1, feature_dim=640):
        aug_features = torch.zeros((n_way, (aug_per_sample + n_shot), feature_dim))
        for cls in range(n_way):
            cls_feature = ori_features[cls, :, :]
            aug_cls_feature = cls_feature + torch.randn(aug_per_sample, feature_dim).cuda() * 0.2
            aug_cls_feature = torch.cat((aug_cls_feature, cls_feature))
            aug_features[cls, :, :] = aug_cls_feature
        return aug_features.cuda()

    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
    
        


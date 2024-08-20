import time
from tqdm import tqdm
import os 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random 
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from util.loss_torch import bpr_loss, l2_reg_loss, contrastLoss
from util.init import *
import torch.nn.init as init 
from base.main_recommender import GraphRecommender
from util.evaluation import early_stopping
from util.sampler import next_batch_unified
from base.torch_interface import TorchGraphInterface
from model.layers import MLP
from model.layers import wavelet

import torch_scatter

#import logging
import sys

#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())
### EDHNN ###
class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, ncount, mcount, mlp1_layers=1, mlp2_layers=1,
        mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False, hypergraph=None, data=None):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP.MLP(in_features, out_features, out_features, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm).to(device)
        else:
            self.W1 = nn.Identity().to(device)

        if mlp2_layers > 0:
            self.W2 = MLP.MLP(in_features+out_features, out_features, out_features, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm).to(device)
        else:
            self.W2 = lambda X: X[..., in_features:].to(device)

        if mlp3_layers > 0:
            self.W = MLP.MLP(out_features, out_features, out_features, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm).to(device)
        else:
            self.W = nn.Identity().to(device)
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout
        self.data = data
        self.device = device
        self.hwnn_args = {
            'filters': out_features,
            'dropout': 0.01,
            'ncount': ncount,
            'mcount': mcount,
            'feature_number': out_features
        }
        self.hwnn_layers = nn.ModuleList([wavelet.HWNN(self.hwnn_args['filters'], self.hwnn_args['dropout'], self.hwnn_args['ncount'], self. hwnn_args['mcount'], self.hwnn_args['feature_number'], self.device, self.data) for i in range(2)])
        self.hgcn_layers = nn.ModuleList([HGCNConv(0.2) for i in range(2)])
        self.mean_pooling = nn.AdaptiveAvgPool1d(out_features)
        self.lns = torch.nn.ModuleList()
        for i in range(2):
            self.lns.append(torch.nn.LayerNorm(out_features))


    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()
    
    def forward(self, X, sparse_norm_adj, X0, act=True):
        N = X.shape[-2]# 2708(node_num)
        beta = 0.9
        '''
        X.shape: [2708, 256] [node_num, feature_dim]
        vetex:tensor([   0,    0,    0,  ..., 1888, 1889, 1890], device='cuda:0')
        edges:tensor([16668, 16755, 16787,  ..., 28952, 16803, 29811], device='cuda:0')
        '''   
        Xve = self.W1(X) # [nnz, C]([7494, 256]) # self.W1(X): [node_num, 256]
        #Xe = self.hwnn_layers[0](Xve, sparse_norm_adj, 'simple_msg_e')
        
        Xe = self.lns[0](self.hgcn_layers[0](sparse_norm_adj, Xve, act=True)) + Xve
        # edge gathering
        Xev = self.W2(torch.cat([X, Xe], -1))
        Xev = self.mean_pooling(Xev)
        X_v = self.lns[1](self.hgcn_layers[1](sparse_norm_adj, Xev, act=True)) + Xev

        X = X_v
        X = (1-self.alpha) * X + self.alpha * X0
        X = self.W(X)
        
        return X
    
    # def forward(self, X, vertex, edges, sparse_norm_adj, X0, act=True):
    #     N = X.shape[-2]# 2708(node_num)
    #     '''
    #     X.shape: [2708, 256] [node_num, feature_dim]
    #     vetex:tensor([   0,    0,    0,  ..., 1888, 1889, 1890], device='cuda:0')
    #     edges:tensor([16668, 16755, 16787,  ..., 28952, 16803, 29811], device='cuda:0')
    #     '''
    
    #     Xve = self.W1(X)[..., vertex, :] # [nnz, C]
    #     Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        

    #     Xev = Xe[..., edges, :] # [nnz, C]
    #     Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
    #     Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

    #     X = Xv
       
    #     X = (1-self.alpha) * X + self.alpha * X0
    #     X = self.W(X)

    #     return X

class HGCNConv(nn.Module):
    def __init__(self, leaky):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embs, act=True):
        if act:
            # print(f"adj.shape: {adj.shape}")
            # print(f"embs.shape: {embs.shape}")
            return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)))
        else:
            # print(f"adj.shape: {adj.shape}")
            # print(f"embs.shape: {embs.shape}")
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

    '''---'''
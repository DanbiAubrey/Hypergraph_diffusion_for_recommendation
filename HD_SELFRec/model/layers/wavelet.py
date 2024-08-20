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

import torch_scatter
import torch.sparse as sparse
from model.layers.wavelettransform import WaveletTransform 

#import logging
import sys
#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class HWNN(torch.nn.Module):
    def __init__(self, filters, dropout, ncount, mcount, feature_number, device, data):
        super(HWNN, self).__init__()
        # self.features=features
        self.ncount = ncount
        self.mcount = mcount
        self.feature_number = feature_number
        self.filters = filters
        self.device = device
        self.data = data
        self.dropout = dropout
        self.setup_layers()

    def setup_layers(self):
        self.convolution_1 = HWNNLayer(self.feature_number,
                                       self.filters,
                                       self.ncount,
                                       self.mcount,
                                       self.device,
                                       K1=3,
                                       K2=3,
                                       approx=True,
                                       data=self.data)

        # self.convolution_2 = HWNNLayer(self.filters,
        #                                self.filters,
        #                                self.ncount,
        #                                self.device,
        #                                K1=3,
        #                                K2=3,
        #                                approx=True,
        #                                data=self.data)

    def forward(self, features, wavelet, msg):
        features = features.to(self.device)
        channel_feature = []
        
        deep_features_1 = self.convolution_1(features, wavelet, msg)
        #deep_features_1 = F.relu(self.convolution_1(features,self.data, msg))
        # deep_features_1 = F.dropout(deep_features_1, self.dropout)
        # deep_features_2 = self.convolution_2(deep_features_1,
        #                                         self.data, msg)
        # channel_feature.append(deep_features_2)
        
        channel_feature.append(deep_features_1)
        return channel_feature[-1]

class HWNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ncount, mcount, device, K1=2, K2=2, approx=True, data=None):
        super(HWNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.ncount = ncount
        self.data = data
        self.K1 = K1
        self.K2 = K2
        self.ncount = self.data.n_users + self.data.n_items
        self.mcount = mcount
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.approx = approx
        self.ui_adj = self.data.ui_adj
        self.hyper_uu = torch.tensor(self.ui_adj.todense()[:self.data.n_users, self.data.n_users:], requires_grad=False).to(device)# sparse_norm_adj: [user+item, user+item]
        self.hyper_ii = torch.tensor(self.ui_adj.todense()[self.data.n_users, :self.data.n_users], requires_grad=False).to(device)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).to(device)
        # self.wavelet = WaveletTransform(data.ui_adj, approx=True)
        # self.wavelet_hypergraph = self.wavelet.generate_hypergraph()
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, wavelet_hypergraph, msg):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(device)
        features = features.to(device)

        if msg == "msg_e":
            Theta = wavelet_hypergraph["E_Theta"].to(device).to_sparse()
            Theta_t = torch.transpose(Theta, 0, 1)
        elif msg == "msg_v":
            Theta = wavelet_hypergraph["V_Theta"].to(device).to_sparse()
            Theta_t = torch.transpose(Theta, 0, 1)
        elif msg == "simple_msg_e":
            Theta = self.sparse_norm_adj.t()
            Theta_t = torch.transpose(Theta, 0, 1)
        elif msg == "simple_msg_v":
            Theta = self.sparse_norm_adj
            Theta_t = torch.transpose(Theta, 0, 1)
        else:
            Theta = torch.sparse.mm(self.sparse_norm_adj, self.sparse_norm_adj.t())
            Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            eye_ncount = torch.eye(self.ncount, device=device).to_sparse()
            poly = self.par[0] * eye_ncount
            Theta_mul = torch.eye(self.ncount).to(device).to_sparse()
            for ind in range(1, self.K1):
                Theta_mul = torch.sparse.mm(Theta_mul, Theta).to_sparse()
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * eye_ncount
            Theta_mul = eye_ncount
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = torch.sparse.mm(Theta_mul, Theta_t)
                poly_t = poly_t + self.par[ind] * Theta_mul
            # poly=self.par[0]*torch.eye(self.ncount).to(self.device)+self.par[1]*Theta+self.par[2]*Theta@Theta
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta_t + self.par[5] * Theta_t @ Theta_t
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta + self.par[
            #     5] * Theta @ Theta

            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            print("wavelets!")
            wavelets = self.wavelet_hypergraph["wavelets"].to(device)
            wavelets_inverse = self.wavelet_hypergraph["wavelets_inv"].to(device)
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix
    
        return local_fea_1


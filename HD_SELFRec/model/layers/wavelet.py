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

#import logging
import sys
#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class HWNN(torch.nn.Module):
    def __init__(self, filters, dropout, ncount, feature_number, device, data):
        super(HWNN, self).__init__()
        # self.features=features
        self.ncount = ncount
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

    def forward(self, features, msg):
        features = features.to(self.device)
        channel_feature = []
        
        deep_features_1 = self.convolution_1(features, self.data, msg)
        #deep_features_1 = F.relu(self.convolution_1(features,self.data, msg))
        # deep_features_1 = F.dropout(deep_features_1, self.dropout)
        # deep_features_2 = self.convolution_2(deep_features_1,
        #                                         self.data, msg)
        # channel_feature.append(deep_features_2)
        
        channel_feature.append(deep_features_1)
        return channel_feature[-1]

class HWNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ncount, device, K1=2, K2=2, approx=True, data=None):
        super(HWNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.ncount = ncount
        self.K1 = K1
        self.K2 = K2
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.ncount = data.n_users + data.n_items
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.approx = approx
        self.wavelet = WaveletTransform(data.ui_adj, approx=True)
        self.wavelet_hypergraph = self.wavelet.generate_hypergraph()
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, data, msg):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter).to(device)
        features = features.to(device)

        if msg == "msg_e":
            Theta = self.wavelet_hypergraph["E_Theta"].to(device).to_sparse()
            Theta_t = torch.transpose(Theta, 0, 1)
        elif msg == "msg_v":
            Theta = self.wavelet_hypergraph["V_Theta"].to(device).to_sparse()
            Theta_t = torch.transpose(Theta, 0, 1)
        else:
            Theta = self.wavelet_hypergraph["Theta"].to(device).to_sparse()
            Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            poly = self.par[0] * torch.eye(self.ncount).to(device).to_sparse()
            Theta_mul = torch.eye(self.ncount).to(device).to_sparse()
            for ind in range(1, self.K1):
                Theta_mul = torch.sparse.mm(Theta_mul, Theta).to_sparse()
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.ncount).to_sparse().to(device)
            Theta_mul = torch.eye(self.ncount).to_sparse().to(device)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = torch.sparse.mm(Theta_mul, Theta_t).to_sparse()
                poly_t = poly_t + self.par[ind] * Theta_mul
            # poly=self.par[0]*torch.eye(self.ncount).to(self.device)+self.par[1]*Theta+self.par[2]*Theta@Theta
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta_t + self.par[5] * Theta_t @ Theta_t
            # poly_t = self.par[3] * torch.eye(self.ncount).to(self.device) + self.par[4] * Theta + self.par[
            #     5] * Theta @ Theta
            # print(poly.shape)#
            # print(diagonal_weight_filter.shape)
            # print(features.shape)
            # print(self.weight_matrix.shape)
            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
            #logging.debug("local feature generated")
        else:
            print("wavelets!")
            wavelets = self.wavelet_hypergraph["wavelets"].to(device)
            wavelets_inverse = self.wavelet_hypergraph["wavelets_inv"].to(device)
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix
        
            localized_features = local_fea_1
            # print(localized_features)
            # print(localized_features.shape)
        #print(f"local_fea_1.shape: {local_fea_1.shape}")
        return local_fea_1

class WaveletTransform:
    # https://github.com/sheldonresearch/HWNN/blob/master/data.py
    def __init__(self, hypergraph, approx):
        self.indice_matrix = torch.tensor(hypergraph.todense())
        self.s = 1.0
        self.approx = approx
        self.device = device

    def generate_hypergraph(self):
        W_e_diag = torch.ones(self.indice_matrix.size()[1])
        D_e_diag = torch.sum(self.indice_matrix.to_dense(), 0)
        D_e_diag = D_e_diag.view((D_e_diag.size()[0]))  # torch.Size([16668])

        D_v_diag = self.indice_matrix.to_dense().mm(W_e_diag.view((W_e_diag.size()[0]), 1))
        D_v_diag = D_v_diag.view((D_v_diag.size()[0]))  # torch.Size([16668])

        #Theta for original HWNN
        Theta = torch.sparse.mm(torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse(),
                torch.sparse.mm(self.indice_matrix,
                torch.sparse.mm(torch.diag(W_e_diag).to_sparse(),
                torch.sparse.mm(torch.diag(torch.pow(D_e_diag, -1)).to_sparse(),
                torch.sparse.mm(self.indice_matrix.t(),
                torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse())))))

        # Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
        #         self.indice_matrix @ torch.diag(W_e_diag) @ \
        #         torch.diag(torch.pow(D_e_diag, -1)) @ \
        #         torch.transpose(self.indice_matrix, 0, 1) @ \
        #         torch.diag(torch.pow(D_v_diag, -0.5))#torch.Size([16668, 16668])

        Theta = Theta.to_dense()
        #Theta = Theta.to_dense()
        Theta_inverse = torch.pow(Theta, -1)
        Theta_inverse[Theta_inverse == float("Inf")] = 0

        # Theta_I = torch.sparse.mm(torch.diag(W_e_diag + torch.ones_like(W_e_diag)).to_sparse(),
        #           torch.sparse.mm(torch.diag(torch.pow(D_e_diag, -1)).to_sparse(),
        #           torch.sparse.mm(self.indice_matrix.t(),
        #           torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse())))

        # Theta_I = Theta_I.to_dense()
        # Theta_I[Theta_I != Theta_I] = 0  # NaN to 0
        # Theta_I_inverse = torch.pow(Theta_I, -1)
        # Theta_I_inverse[Theta_I_inverse == float("Inf")] = 0

        # Theta for EDHNN
        E_Theta = torch.sparse.mm(torch.diag(W_e_diag).to_sparse(),
                torch.sparse.mm(torch.diag(torch.pow(D_e_diag, -1)).to_sparse(),
                torch.sparse.mm(self.indice_matrix.t(),
                torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse())))

        E_Theta = E_Theta.to_dense()

        # E_Theta_I = torch.sparse.mm(torch.diag(W_e_diag + torch.ones_like(W_e_diag)).to_sparse(),
        #         torch.sparse.mm(torch.diag(torch.pow(D_e_diag, -1)).to_sparse(),
        #         torch.sparse.mm(self.indice_matrix.t(),
        #        torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse())))

        # E_Theta_I = E_Theta_I.to_dense()

        # V_Theta = torch.sparse.mm(torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse(), self.indice_matrix)

        # V_Theta = V_Theta.to_dense()

        # V_Theta_I = torch.sparse.mm(torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse(), self.indice_matrix)

        # V_Theta_I = V_Theta_I.to_dense()

        Laplacian = torch.eye(Theta.size()[0]) - Theta

        wavelets = []
        wavelets_inv = []
        wavelets_t = []
        fourier_e = []
        fourier_v = []

        if self.approx == False:
            fourier_e, fourier_v = torch.linalg.eigh(Laplacian, UPLO='U')

            wavelets = torch.matmul(fourier_v, torch.matmul(torch.diag(torch.exp(-1.0 * fourier_e * self.s)), fourier_v.t()))
            wavelets_inv = torch.matmul(fourier_v, torch.matmul(torch.diag(torch.exp(fourier_e * self.s)), fourier_v.t()))
            wavelets_t = wavelets.t()

            wavelets[wavelets < 0.00001] = 0
            wavelets_inv[wavelets_inv < 0.00001] = 0
            wavelets_t[wavelets_t < 0.00001] = 0

        hypergraph = {
            "indice_matrix": self.indice_matrix.to_dense(),
            "D_v_diag": D_v_diag,
            "D_e_diag": D_e_diag,
            "W_e_diag": W_e_diag,  # hyperedge_weight_flat
            "laplacian": Laplacian.to_dense(),
            "fourier_v": fourier_v,
            "fourier_e": fourier_e,
            "wavelets": wavelets,
            "wavelets_inv": wavelets_inv,
            "wavelets_t": wavelets_t,
            "Theta": Theta.to_dense(),
            "E_Theta": E_Theta.to_dense()
            # "V_Theta": V_Theta.to_dense()
            
        }
        return hypergraph


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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class WaveletTransform:
    # https://github.com/sheldonresearch/HWNN/blob/master/data.py
    def __init__(self, hypergraph, approx):
        self.indice_matrix = torch.tensor(hypergraph.to_dense())
        self.s = 1.0
        self.approx = approx
        self.device = device

    def generate_hypergraph(self):
        W_e_diag = torch.ones(self.indice_matrix.size()[1]).to(device)
        D_e_diag = torch.sum(self.indice_matrix.to_dense(), 0).to(device)
        D_e_diag = D_e_diag.view((D_e_diag.size()[0]))  # torch.Size([16668])

        D_v_diag = self.indice_matrix.to_dense().mm(W_e_diag.view((W_e_diag.size()[0]), 1)).to(device)
        D_v_diag = D_v_diag.view((D_v_diag.size()[0]))  # torch.Size([16668])

        #Theta for original HWNN
        # Theta = torch.sparse.mm(torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse(),
        #         torch.sparse.mm(self.indice_matrix,
        #         torch.sparse.mm(torch.diag(W_e_diag).to_sparse(),
        #         torch.sparse.mm(torch.diag(torch.pow(D_e_diag, -1)).to_sparse(),
        #         torch.sparse.mm(self.indice_matrix.t(),
        #         torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse())))))

        Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
                self.indice_matrix @ torch.diag(W_e_diag) @ \
                torch.diag(torch.pow(D_e_diag, -1)) @ \
                torch.transpose(self.indice_matrix, 0, 1) @ \
                torch.diag(torch.pow(D_v_diag, -0.5))#torch.Size([16668, 16668])

        #Theta = Theta.to_dense().to(device)
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

        E_Theta = E_Theta.to(device)

        # E_Theta_I = torch.sparse.mm(torch.diag(W_e_diag + torch.ones_like(W_e_diag)).to_sparse(),
        #         torch.sparse.mm(torch.diag(torch.pow(D_e_diag, -1)).to_sparse(),
        #         torch.sparse.mm(self.indice_matrix.t(),
        #        torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse())))

        # E_Theta_I = E_Theta_I.to_dense()

        # V_Theta = torch.sparse.mm(torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse(), self.indice_matrix)

        # V_Theta = V_Theta.to_dense()

        # V_Theta_I = torch.sparse.mm(torch.diag(torch.pow(D_v_diag, -0.5)).to_sparse(), self.indice_matrix)

        # V_Theta_I = V_Theta_I.to_dense()

        # Laplacian = torch.eye(Theta.size()[0]) - Theta

        # wavelets = []
        # wavelets_inv = []
        # wavelets_t = []
        # fourier_e = []
        # fourier_v = []

        # if self.approx == False:
        #     fourier_e, fourier_v = torch.linalg.eigh(Laplacian, UPLO='U')

        #     wavelets = torch.matmul(fourier_v, torch.matmul(torch.diag(torch.exp(-1.0 * fourier_e * self.s)), fourier_v.t()))
        #     wavelets_inv = torch.matmul(fourier_v, torch.matmul(torch.diag(torch.exp(fourier_e * self.s)), fourier_v.t()))
        #     wavelets_t = wavelets.t()

        #     wavelets[wavelets < 0.00001] = 0
        #     wavelets_inv[wavelets_inv < 0.00001] = 0
        #     wavelets_t[wavelets_t < 0.00001] = 0

        hypergraph = {
            "indice_matrix": self.indice_matrix.to_dense(),
            "D_v_diag": D_v_diag,
            "D_e_diag": D_e_diag,
            "W_e_diag": W_e_diag,  # hyperedge_weight_flat
            # "laplacian": Laplacian.to_dense(),
            # "fourier_v": fourier_v,
            # "fourier_e": fourier_e,
            # "wavelets": wavelets,
            # "wavelets_inv": wavelets_inv,
            # "wavelets_t": wavelets_t,
            "Theta": Theta.to_sparse(),
            "E_Theta": E_Theta
            # "V_Theta": V_Theta.to_dense()
            
        }
        return hypergraph


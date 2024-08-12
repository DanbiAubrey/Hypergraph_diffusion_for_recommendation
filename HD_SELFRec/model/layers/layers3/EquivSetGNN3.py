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
from model.layers.layers3 import EquivSetConv3

import torch_scatter
#import logging
import sys
#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())
class EquivSetGNN3(nn.Module):
    def __init__(self, num_features, args, dense_hypergraph, data):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        nhid = args['MLP_hidden']
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args['activation']]
        self.input_drop = nn.Dropout(args['input_dropout']) # 0.6 is chosen as default
        self.dropout = nn.Dropout(args['dropout']) # 0.2 is chosen for GCNII
        self.data = data

        self.in_channels = num_features
        self.hidden_channels = args['MLP_hidden']
        #self.output_channels = num_classes

        self.mlp1_layers = args['MLP_num_layers']
        self.mlp2_layers = args['MLP_num_layers'] if args['MLP2_num_layers'] < 0 else args['MLP2_num_layers']
        self.mlp3_layers = args['MLP_num_layers'] if args['MLP3_num_layers'] < 0 else args['MLP3_num_layers']
        self.nlayer = args['All_num_layers']

        self.lin_in = torch.nn.Linear(num_features, args['MLP_hidden'])
        
        self.conv = EquivSetConv3.EquivSetConv3(args['MLP_hidden'], args['MLP_hidden'], mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers, alpha=args['restart_alpha'], aggr=args['aggregate'],
            dropout=args['dropout'], normalization=args['normalization'], input_norm=args['AllSet_input_norm'], hypergraph=dense_hypergraph, data=self.data)
        
        # self.classifier = MLP(in_channels=args.MLP_hidden,
        #     hidden_channels=args.Classifier_hidden,
        #     out_channels=num_classes,
        #     num_layers=args.Classifier_num_layers,
        #     dropout=args.dropout,
        #     Normalization=args.normalization,
        #     InputNorm=False)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()
        
        #self.classifier.reset_parameters()

    def forward(self, x, hypergraph, n_nodes):
        #x = data.x # x: [num_node, latent]
        #V, E = data.edge_index[0], data.edge_index[1] # V: 7494(unique_values:2708), E: 7494(unique_values:4287)
        
        '''my code'''
        # build bipartite graphs for user and item(star expension)
        V, E = self.generate_V_E(n_nodes, hypergraph)
        '''---'''
        lamda, alpha = 0.5, 0.1
        x = self.dropout(x)
        x = F.relu(self.lin_in(x))
        x0 = x
        for i in range(self.nlayer):
            #logging.debug("propagating through edhnn layers...")
            x = self.dropout(x)
            #x = self.conv(x, V, E, x0)
            #logging.debug("conv layer...")
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        #x = self.classifier(x)
        return x

    '''my_code'''
    def generate_V_E(self, n_nodes, hypergraph):
        
        new_connections = self.build_new_hypergraph(hypergraph)

        non_zero_indices = torch.nonzero(hypergraph > 0)
        n_connections = non_zero_indices.size(0)
        
        vertex_n = hypergraph.shape[0]
        edge_n = hypergraph.shape[1]

        V = torch.zeros(n_connections, dtype=torch.long)# [120938]
        E = torch.zeros(n_connections, dtype=torch.long)# [120938]
        
        V = non_zero_indices[:, 0].to(device)
        E = non_zero_indices[:, 1].to(device)
        
        # print(V.device)
        # print(E.device)
        
        # idx = 0
        # for n in range(hypergraph.size(0)):
        #     for e in range(hypergraph.size(1)):
        #         element = hypergraph[n,e]
        #         if element > 0:
        #             V[idx] = n
        #             E[idx] = e + n_nodes
        #             idx += 1

        return V, E

    def build_new_hypergraph(self, hypergraph):

        vertex_n = hypergraph.shape[0]
        edge_n = hypergraph.shape[1]

        list_vertices = torch.arange(vertex_n, dtype= torch.long)#16668
        list_edges = torch.arange(edge_n, dtype = torch.long) + vertex_n
        #tensor([16668, 16669, 16670,  ..., 33333, 33334, 33335])

        new_n_vertex = vertex_n + edge_n
        new_vertices = torch.cat((list_vertices,list_edges))
       
        #new E-> V connections
        V_E_connections = torch.nonzero(hypergraph > 0)
        E_V_connections = V_E_connections
        E_V_connections = E_V_connections[:, [1, 0]]
        E_V_connections[:,0] += vertex_n
       
        new_connections = torch.cat((V_E_connections, E_V_connections), 0)
        
        return new_connections
        
import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
import time 

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.evaluation import early_stopping

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class DHCF_diffusion(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(DHCF_diffusion, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self.kwargs = kwargs
        self.model = DHCF_Encoder(self.config, self.data, kwargs)
        self._parse_config(self.config, kwargs)
        
    def _parse_config(self, config, kwargs):
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.reg = float(kwargs['reg'])
        self.latent_size = int(kwargs['embedding_size'])
        self.hyperDim = int(kwargs['hyper_dim'])
        self.drop_rate = float(kwargs['drop_rate'])
        self.leaky = float(kwargs['p'])
        self.nLayers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
        self.temp = float(kwargs['temp'])
        self.seed = int(kwargs['seed'])
        self.weight_decay =float(kwargs['weight_decay'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        
    def train(self, load_pretrained):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.lr_decay, patience=5)

        lst_train_losses = []
        lst_rec_losses = []
        lst_reg_losses = []
        lst_performances = []
        recall_list = []
        
        for epoch in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []
            
            s_train = time.time()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) 
                reg_loss =  l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb) /self.batch_size
                batch_loss = rec_loss + reg_loss

                train_losses.append(batch_loss.item())
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            
            e_train = time.time() 
            tr_time = e_train - s_train 

            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)

            lst_train_losses.append([epoch, train_loss])
            lst_rec_losses.append([epoch,rec_loss])
            lst_reg_losses.append([epoch, reg_loss])
            scheduler.step(train_loss)

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                cur_data, data_ep = self.fast_evaluation(epoch, train_time=tr_time)
                lst_performances.append(data_ep)
                
                cur_recall =  float(cur_data[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                if should_stop:
                    break 
        
        self.save_loss(lst_train_losses, lst_rec_losses, lst_reg_losses)
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            print("Saving")
            self.save_model(self.model)     

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class HGCNConv(nn.Module):
    def __init__(self, leaky):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embs, act=True):
        if act:
            return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)))
        else:
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

class DHCF_Encoder(nn.Module):
    def __init__(self, config, data, args):
        super(DHCF_Encoder, self).__init__()
        self.data = data
        adj = data.interaction_mat
        self.adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(adj).to_dense().to(device)#[n_user, n_item]

        self._parse_args(args)
        self.embedding_dict = self._init_model()
        
        self.fc_u = nn.Linear(self.hyper_dim, self.hyper_dim)
        self.fc_i = nn.Linear(self.hyper_dim, self.hyper_dim)
        
        # self.hgnn_u =  [HGCNConv(leaky=self.p) for _ in range(self.layers)]
        # self.hgnn_i = [HGCNConv(leaky=self.p) for _ in range(self.layers)]
        '''my code'''
        self.edhnn_user_n = self.data.n_users + self.n_edges # 2019 #user_n = 1890
        self.edhnn_item_n = self.data.n_items + self.n_edges # 14905 #item_n = 14777

        self.edgnn_u =  [EquivSetGNN(self.latent_size, self.edhnn_args) for _ in range(self.layers)]#todo: put arguments
        self.edgnn_i = [EquivSetGNN(self.latent_size, self.edhnn_args) for _ in range(self.layers)]#todo: put arguments
        '''------'''
        self.non_linear = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_rate)
        
    def _parse_args(self, args):
        self.input_dim = args['input_dim']
        self.hyper_dim = args['hyper_dim']
        self.p = args['p']
        self.drop_rate = args['drop_rate'] 
        self.layers = args['n_layers']
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.hyper_dim)).to(device)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.hyper_dim)).to(device))
        })
        return embedding_dict

    def forward(self):
        uEmbed = self.embedding_dict['user_emb']
        iEmbed = self.embedding_dict['item_emb']
        
        user_embeds = [uEmbed]
        item_embeds = [iEmbed]
        
        for idx, layer in enumerate(range(self.layers)):
            hyper_u_embed = self.edgnn_u[idx](self.adj, uEmbed)
            hyper_i_embed = self.edgnn_i[idx](self.adj.T, iEmbed)
            
            user_embeds.append(hyper_u_embed)
            item_embeds.append(hyper_i_embed)
        
        user_all_embeddings = torch.cat(user_embeds, dim=1)
        item_all_embeddings = torch.cat(item_embeds, dim=1)
        
        return user_all_embeddings, item_all_embeddings 

### EDHNN ###

class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
        mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm).to(device)
        else:
            self.W1 = nn.Identity().to(device)

        if mlp2_layers > 0:
            self.W2 = MLP(in_features+out_features, out_features, out_features, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm).to(device)
        else:
            self.W2 = lambda X: X[..., in_features:].to(device)

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm).to(device)
        else:
            self.W = nn.Identity().to(device)
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]# 2708(node_num)
        # X.shape: [2708, 256] [node_num, feature_dim]

        #select the processed nodes
        Xve = self.W1(X)[..., vertex, :] # [nnz, C]([7494, 256]) # self.W1(X): [node_num, 256]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        
        Xev = Xe[..., edges, :] # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

        X = Xv

        X = (1-self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

class EquivSetGNN(nn.Module):
    def __init__(self, num_features, args):
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

        self.in_channels = num_features
        self.hidden_channels = args['MLP_hidden']
        #self.output_channels = num_classes

        self.mlp1_layers = args['MLP_num_layers']
        self.mlp2_layers = args['MLP_num_layers'] if args['MLP2_num_layers'] < 0 else args['MLP2_num_layers']
        self.mlp3_layers = args['MLP_num_layers'] if args['MLP3_num_layers'] < 0 else args['MLP3_num_layers']
        self.nlayer = args['All_num_layers']

        self.lin_in = torch.nn.Linear(num_features, args['MLP_hidden'])
        
        self.conv = EquivSetConv(args['MLP_hidden'], args['MLP_hidden'], mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers, alpha=args['restart_alpha'], aggr=args['aggregate'],
            dropout=args['dropout'], normalization=args['normalization'], input_norm=args['AllSet_input_norm'])
        
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
            x = self.dropout(x)
            #x = self.conv(x, V, E, x0)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        #x = self.classifier(x)
        return x

    '''my_code'''
    def generate_V_E(self, n_nodes, hypergraph):
        
        non_zero_indices = torch.nonzero(hypergraph > 0)
        n_connections = non_zero_indices.size(0)
        
        V = torch.zeros(n_connections, dtype=torch.long).to(device)# [120938]
        E = torch.zeros(n_connections, dtype=torch.long).to(device)# [120938]
        
        V = non_zero_indices[:, 0]
        E = non_zero_indices[:, 1] + n_nodes
        
        # idx = 0
        # for n in range(hypergraph.size(0)):
        #     for e in range(hypergraph.size(1)):
        #         element = hypergraph[n,e]
        #         if element > 0:
        #             V[idx] = n
        #             E[idx] = e + n_nodes
        #             idx += 1

        return V, E
    '''---'''
###### ------ ######

###### MLP ######
class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels # first normalization
        flops += num_samples * self.in_channels * self.hidden_channels # first linear layer
        flops += num_samples * self.hidden_channels # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += num_samples * self.out_channels * self.hidden_channels # last linear layer

        return flops
###### ------ ######

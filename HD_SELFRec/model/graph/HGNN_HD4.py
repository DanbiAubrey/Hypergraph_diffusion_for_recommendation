import torch
import torch.nn as nn
import numpy as np 
import pandas as pd 
import time
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, contrastLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.evaluation import early_stopping
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
from model.layers.layers2 import EquivSetGNN2
from model.layers.wavelet import WaveletTransform

'''
python main.py --model=HGNN_HD4 --dataset=lastfm  --lrate=0.0001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20 --mode=local_only
python main.py --model=HGNN_HD4 --dataset=amazon_books  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.001 --early_stopping_steps=20 --seed=20 --mode=local_only --n_layers=3
python main.py --model=HGNN_HD4 --dataset=steam  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=HGNN_HD4 --dataset=yelp  --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'# address cuda overload
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20\
class HGNN_HD4(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        GraphRecommender.__init__(self, conf, training_set, test_set, knowledge_set, **kwargs)
        self.device = device
        self._parse_config( kwargs)
        self.set_seed()
        self.model = HGNNModel(self.data, kwargs, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lRate, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.lr_decay, patience=10)


    def _parse_config(self, kwargs):
        self.dataset = kwargs['dataset']
        self.lRate = float(kwargs['lrate'])
        self.lr_decay = float(kwargs['lr_decay'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.batchSize = int(kwargs['batch_size'])
        self.batchSizeKG = int(kwargs['batch_size_kg'])
        self.reg = float(kwargs['reg'])
        self.reg_kg = float(kwargs['reg_kg'])
        self.hyper_dim = int(kwargs['hyper_dim'])
        self.p = float(kwargs['p'])
        self.drop_rate = float(kwargs['drop_rate'])
        self.layers = int(kwargs['n_layers'])
        self.cl_rate = float(kwargs['cl_rate'])
        self.temp = kwargs['temp']
        self.seed = kwargs['seed']
        self.mode = kwargs['mode']
        self.early_stopping_steps = kwargs['early_stopping_steps']
        self.weight_decay = kwargs['weight_decay']

        if self.mode == 'full':
            self.use_contrastive = True
            self.use_local = True
            self.use_group = True
        elif self.mode == 'group_only':
            self.use_contrastive = False
            self.use_local = False
            self.use_gruop = True
        elif self.mode == 'local_only':
            self.use_contrastive = False
            self.use_local = True
            self.use_group = False
        else:
            self.use_contrastive = True
            self.use_local = True
            self.use_group = True
            
    def set_seed(self):
        seed = self.seed 
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

    def train(self, load_pretrained=False):
        print("start training")
        train_model = self.model 
        lst_train_losses = []
        lst_cf_losses = []
        lst_cl_losses = [] 
        lst_performances = []
        recall_list = []
                
        lst_performances = []
        recall_list = []

        for ep in range(self.maxEpoch):
            cf_losses = []
            cl_losses = [] 

            cf_total_loss = 0
            cl_total_loss = 0 

            n_cf_batch = int(self.data.n_cf_train // self.batchSize + 1)

            train_model.train()
            s_train = time.time()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                if self.mode == 'local_only':
                    user_emb_lc, item_emb_lc = train_model(mode='local', keep_rate=1-self.drop_rate)
                
                    anchor_emb = user_emb_lc[user_idx]
                    pos_emb = item_emb_lc[pos_idx]
                    neg_emb = item_emb_lc[neg_idx]
                elif self.mode == 'group_only':
                    user_emb_grp, item_emb_grp = train_model(mode='group', keep_rate=1-self.drop_rate)
                    
                    anchor_emb = user_emb_grp[user_idx]
                    pos_emb = item_emb_grp[pos_idx]
                    neg_emb = item_emb_grp[neg_idx]
                else:
                    user_emb_lc, item_emb_lc = train_model(mode='local', keep_rate=1-self.drop_rate)
                    user_emb_grp, item_emb_grp = train_model(mode='group', keep_rate=1-self.drop_rate)
                    
                    user_emb_fused = torch.mean(torch.stack([user_emb_lc, user_emb_grp], dim=1), dim=1)
                    item_emb_fused = torch.mean(torch.stack([item_emb_lc, item_emb_grp], dim=1), dim=1)
                    anchor_emb = user_emb_fused[user_idx]
                    pos_emb = item_emb_fused[pos_idx]
                    neg_emb = item_emb_fused[neg_idx]

                    h_local = torch.cat([user_emb_lc, item_emb_lc], dim=0)
                    h_group = torch.cat([user_emb_grp, item_emb_grp], dim=0)
                
                cf_batch_loss = train_model.calculate_cf_loss(anchor_emb, pos_emb, neg_emb, self.reg)
                cf_total_loss += cf_batch_loss.item()

                if self.use_contrastive:
                    cl_batch_loss = self.cl_rate * train_model.calculate_ssl_loss(self.data, user_idx, pos_idx, h_local, h_group, self.temp)
                    cl_losses.append(cl_batch_loss.item())
                    cl_total_loss += cl_batch_loss.item()
                    batch_loss = cf_batch_loss + cl_batch_loss
                else:
                    cl_batch_loss = 0
                    batch_loss = cf_batch_loss

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                cf_losses.append(cf_batch_loss.item())

                if (n % 20) == 0:
                    if self.use_contrastive:
                        print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
                        print('CL Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch, cl_batch_loss.item(), cl_total_loss / (n+1)))
                    else:
                        print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(ep, n, n_cf_batch,  cf_batch_loss.item(), cf_total_loss / (n+1)))
                
                cf_loss = np.mean(cf_losses)
                e_train = time.time() 
                train_time = e_train - s_train

                if self.use_contrastive:
                    cl_loss = np.mean(cl_losses)
                    train_loss = cf_loss + cl_loss 
                else:
                    cl_loss  = 0
                    train_loss = cf_loss

                lst_cf_losses.append([ep,cf_loss])
                lst_train_losses.append([ep, train_loss])
                lst_cl_losses.append([ep, cl_loss])

                self.scheduler.step(train_loss)

                # Evaluation
                train_model.eval()

            with torch.no_grad():
                if self.mode == 'local_only':
                    user_emb_lc, item_emb_lc = train_model(mode='local')
                    
                    user_emb = user_emb_lc
                    item_emb = item_emb_lc 
                elif self.mode == 'group_only':
                    user_emb_grp, item_emb_grp = train_model(mode='group')
                    
                    user_emb = user_emb_grp
                    item_emb = item_emb_grp
                else:
                    user_emb_lc, item_emb_lc = train_model(mode='local')
                    user_emb_grp, item_emb_grp = train_model(mode='group')

                    user_emb = torch.mean(torch.stack([user_emb_lc, user_emb_grp], dim=1), dim=1)
                    item_emb = torch.mean(torch.stack([item_emb_lc, item_emb_grp], dim=1), dim=1)

                self.user_emb, self.item_emb = user_emb, item_emb
            
                cur_data, data_ep = self.fast_evaluation(ep, train_time=train_time)
                lst_performances.append(data_ep)
                
                cur_recall =  float(cur_data[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                
                if should_stop:
                    break 
        
            kg_loss = 0

        self.save_loss(lst_train_losses, lst_cf_losses, lst_cl_losses)
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    
    def predict(self, u):
        user_id  = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def calculate_loss(self, anchor_emb, pos_emb, neg_emb, batch_size):
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss = l2_reg_loss(self.reg, anchor_emb, pos_emb, neg_emb) / batch_size
        return rec_loss, reg_loss
    
    def save(self):
        with torch.no_grad():
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()
            if self.mode == 'local_only':
                self.best_user_emb, self.best_item_emb = self.model.forward(mode='local')
            elif self.mode == 'group_only':
                self.best_user_emb, self.best_item_emb = self.model.forward(mode='group')
            else:
                user_emb_lc, item_emb_lc = self.model.forward(mode='local')
                user_emb_grp, item_emb_grp = self.model.forward(mode='group')

                self.best_user_emb = torch.mean(torch.stack([user_emb_lc, user_emb_grp], dim=1), dim=1)
                self.best_item_emb = torch.mean(torch.stack([item_emb_lc, item_emb_grp], dim=1), dim=1)

            self.save_model(self.model)     

class HGNNModel(nn.Module):
    def __init__(self, data, args, device):
        super(HGNNModel, self).__init__()
        self.data = data
        
        self.device = device
        self.use_drop_edge = False
        self.sparse_norm_adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(self.device)

        self.user_indices =  torch.LongTensor(list(data.user.keys())).to(self.device)
        self.item_indices = torch.LongTensor(list(data.item.keys())).to(self.device)

        self._parse_args(args)
        self.embedding_dict = self._init_model()
        self.hgnn_cf = []
    
        self.hgnn_layer_local = LocalAwareEncoder(self.data, self.emb_size, self.hyper_size, self.layers, self.p, self.drop_rate, self.device)
        self.hgnn_layer_group = GroupAwareEncoder(self.data, self.p, self.drop_rate, self.layers, self.emb_size)
       
        self.act = nn.LeakyReLU(self.p)
        self.dropout = nn.Dropout(self.drop_rate)
        self.edgeDropper = SpAdjDropEdge()

    def _parse_args(self, args):
        self.input_dim = args['input_dim']
        self.hyper_dim = args['hyper_dim']
        self.p = args['p']
        self.drop_rate = args['drop_rate'] 
        self.layers = args['n_layers']
        self.temp = args['temp']
        self.aug_type = args['aug_type']
        self.relation_dim = args['relation_dim']
        self.nheads = int(args['nheads'])
        self.emb_size =  int(args['input_dim'])
        self.hyper_size =  int(args['hyper_dim'])
        self.batchSize = int(args['batch_size'])
        self.batchSizeKG = int(args['batch_size_kg'])
        
    def _init_model(self):
        initializer = init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.hyper_dim)).to(self.device)),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.hyper_dim)).to(self.device))
        })
        
        return embedding_dict
    
    def calculate_local_embeddings(self, keep_rate:float=1):
        uEmbed = self.embedding_dict['user_emb']
        iEmbed = self.embedding_dict['item_emb']
        ego_embeddings = torch.cat([uEmbed, iEmbed], 0).to(device)
        sparse_norm_adj = self.edgeDropper(self.sparse_norm_adj, keep_rate)
        user_all_embeddings, item_all_embeddings = self.hgnn_layer_local(ego_embeddings, sparse_norm_adj)
        return user_all_embeddings, item_all_embeddings

    def calculate_group_embeddings(self, keep_rate:float=1):
        uEmbed = self.embedding_dict['user_emb']
        iEmbed = self.embedding_dict['item_emb']
        ego_embeddings = torch.cat([uEmbed, iEmbed], 0).to(device)
        sparse_norm_adj = self.edgeDropper(self.sparse_norm_adj, keep_rate)
        user_all_embeddings, item_all_embeddings = self.hgnn_layer_group(ego_embeddings, sparse_norm_adj)
        return user_all_embeddings, item_all_embeddings

    def forward(self, mode='local', keep_rate=1):
        if mode == 'local':
            user_embed, item_embed = self.calculate_local_embeddings(keep_rate=keep_rate)
            return user_embed, item_embed
        elif mode == 'group':
            entity_embed = self.calculate_group_embeddings(keep_rate=keep_rate)
            return user_embed, item_embed

    def calculate_cf_loss(self, anchor_emb, pos_emb, neg_emb, reg):
        rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
        reg_loss = l2_reg_loss(reg, anchor_emb, pos_emb, neg_emb) / self.batchSize
        cf_loss  = rec_loss + reg_loss
        return cf_loss
    
    def calculate_ssl_loss(self, data, ancs, poss, emb_local, emb_group, temp):
        embeds1 = emb_local
        embeds2 = emb_group
        sslLoss = contrastLoss(embeds1[:data.n_users], embeds2[:data.n_users], torch.unique(ancs), temp) + \
                    contrastLoss(embeds2[data.n_users:], embeds2[data.n_users:], torch.unique(poss), temp)
        return sslLoss
    
class LocalAwareEncoder(nn.Module):
    def __init__(self, data, emb_size, hyper_size, n_layers, leaky, drop_rate, device, use_self_att=False):
        super(LocalAwareEncoder, self).__init__()

        self.data = data
        self.latent_size = emb_size
        self.hyper_size = hyper_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.ui_adj = data.ui_adj
        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(drop_rate)
        self.edgeDropper = SpAdjDropEdge()
        self.sparse_norm_adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(device)
        
        self.edhnn_args = self.init_edhnn_config(self.hyper_size)

        # self.edhnn_layers = nn.ModuleList([EquivSetGNN.EquivSetGNN(hyper_size, self.edhnn_args, self.sparse_norm_adj, self.data) for i in range(2)])
        self.hgcn_layers = nn.ModuleList([HGCNConv(leaky=0.5) for i in range(self.layers)])
        self.edhnn_layers = nn.ModuleList([EquivSetGNN2.EquivSetGNN(hyper_size, self.edhnn_args, self.sparse_norm_adj, self.data) for i in range(self.layers)])
        self.lns = torch.nn.ModuleList()

        for i in range(self.layers):
            self.lns.append(torch.nn.LayerNorm(hyper_size))

        # self.edhnn_user_n, self.edhnn_item_n = self.data.n_users, self.data.n_items # norm_adj:[u_n +i_n, u_n +i_n]
        self.edhnn_ui_n = self.data.n_items + self.data.n_users

        self.hyper_uu = torch.tensor(self.ui_adj.todense()[:self.data.n_users, self.data.n_users:], requires_grad=False).to(device)# sparse_norm_adj: [user+item, user+item]
        self.hyper_ii = torch.tensor(self.ui_adj.todense()[self.data.n_users, :self.data.n_users], requires_grad=False).to(device)

        self.dense_hypergraph = torch.tensor(self.ui_adj.todense())
    
    def init_edhnn_config(self, hyper_size):
        edhnn_args ={
            'MLP_hidden' : hyper_size,
            'MLP1_num_layers' : 0,
            'MLP2_num_layers' : 0,
            'MLP3_num_layers' : 1,
            'MLP_num_layers' : 0,
            'restart_alpha' : 0.0,
            'aggregate' : 'mean',
            'dropout' : 0.5,
            'normalization' : 'ln',
            'input_norm' : True,
            'All_num_layers' : 1,
            'activation' : 'relu',
            'input_dropout': 0.6,
            'AllSet_input_norm': True
        }
        return edhnn_args
    
    def forward(self, ego_embeddings, sparse_norm_adj):
        res = ego_embeddings
        all_embeddings = []

        for k in range(self.layers):
            if k != self.layers - 1: 
                #ego_embeddings = self.lns[k](self.hgcn_layers[0](sparse_norm_adj, ego_embeddings)) + res
                ego_embeddings = self.edhnn_layers[k](ego_embeddings, self.dense_hypergraph, self.edhnn_ui_n) + res
            else:
                ego_embeddings = self.lns[0](self.hgcn_layers[0](sparse_norm_adj, ego_embeddings, act=False)) + res
                #ego_embeddings = self.lns[0](self.edhnn_layers[k](ego_embeddings, self.dense_hypergraph, self.edhnn_ui_n)) + res
            all_embeddings += [ego_embeddings]

        user_all_embeddings = all_embeddings[-1][:self.data.n_users]
        item_all_embeddings = all_embeddings[-1][self.data.n_users:]
        return user_all_embeddings, item_all_embeddings

class GroupAwareEncoder(nn.Module):
    def __init__(self, data, p, drop_rate, layers, emb_size):
        super(GroupAwareEncoder, self).__init__()
        self.layers = layers
        self.data = data
        self.drop_rate = drop_rate
        self.leaky = p
        self.in_channels, self.out_channels = emb_size, emb_size
        self.ncount = self.data.n_users + self.data.n_items
        self.device = device
        #self.gwnn_layers = nn.ModuleList([SparseGraphWaveletLayer(self.in_channels, self.out_channels, self.ncount, self.device) for i in range(self.layers)])
        self.hgcn_layers = nn.ModuleList([HGCNConv(leaky=0.5) for i in range(self.layers)])

    def forward(self, ego_embeddings, sparse_norm_adj):
        res = ego_embeddings
        all_embeddings = []

        for k in range(self.layers):
            if k != self.layers - 1:
                ego_embeddings = self.hgcn_layers[k](sparse_norm_adj,ego_embeddings)
            else:
                ego_embeddings = self.hgcn_layers[k](sparse_norm_adj,ego_embeddings)

        user_all_embeddings = all_embeddings[-1][:self.data.n_users]
        item_all_embeddings = all_embeddings[-1][self.data.n_users:]

        return user_all_embeddings, item_all_embeddings

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

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
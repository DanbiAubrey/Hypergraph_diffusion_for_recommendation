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
#from model.layers.layers2 import EquivSetGNN2
from model.layers import EquivSetGNN
from model.layers.wavelet import WaveletTransform
from model.layers.gwnn_layer import SparseGraphWaveletLayer
from model.layers.wavelettransform import WaveletTransform
from model.layers import wavelet
import torch_scatter

'''
python main.py --model=HGNN_HD3 --dataset=lastfm  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20 --mode=full
python main.py --model=HGNN_HD3 --dataset=amazon_books  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.001 --early_stopping_steps=20 --seed=20 --mode=local_only --n_layers=3
python main.py --model=HGNN_HD3 --dataset=steam  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=HGNN_HD3 --dataset=yelp  --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'# address cuda overload
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20\
class HGNN_HD3(GraphRecommender):
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
        torch.cuda.manual_seed_all(seed)
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
                    # with torch.profiler.profile(
                    #     activities=[
                    #         torch.profiler.ProfilerActivity.CPU, 
                    #         torch.profiler.ProfilerActivity.CUDA  # GPU 사용 시
                    #     ],
                    #     record_shapes=True,  # 각 연산에서 텐서 모양 기록
                    #     profile_memory=True,  # 메모리 사용량 기록
                    #     with_stack=True  # 소스 코드 스택 추적 기록
                    # ) as prof:
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
            #print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
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
            user_embed, item_embed = self.calculate_group_embeddings(keep_rate=keep_rate)
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

        torch.autograd.set_detect_anomaly(True)
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

        self.hgcn_layer = HGCNConv(leaky=0.3)
        self.lns = torch.nn.ModuleList()

        for i in range(self.layers):
            self.lns.append(torch.nn.LayerNorm(hyper_size))

        self.hgnn_layers = nn.ModuleList([HGCNConv(leaky=0.3) for i in range(self.layers)])
        self.edhnn_layers = nn.ModuleList([EquivSetGNN(hyper_size, self.edhnn_args, self.sparse_norm_adj, self.data, self.data.n_users, self.data.n_items) for i in range(self.layers)])
        self.lns = torch.nn.ModuleList()
        for i in range(self.layers):
            self.lns.append(torch.nn.LayerNorm(hyper_size))

        # self.wavelet = WaveletTransform(self.sparse_norm_adj, approx=True)
        # self.wavelet_hypergraph = self.wavelet.generate_hypergraph()

        self.hyper_uu = torch.tensor(self.ui_adj.todense()[:self.data.n_users, self.data.n_users:], requires_grad=False).to(device)# sparse_norm_adj: [user+item, user+item]
        self.hyper_ii = torch.tensor(self.ui_adj.todense()[self.data.n_users:, :self.data.n_users], requires_grad=False).to(device)

        self.edhnn_ui_n = self.data.n_users + self.data.n_items

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
                #ego_embeddings = self.hgnn_layers[k](sparse_norm_adj, ego_embeddings)  + res
                #ego_embeddings = self.lns[k](self.hgnn_layers[k](sparse_norm_adj, ego_embeddings))  + res
                ego_embeddings = self.edhnn_layers[k](ego_embeddings, sparse_norm_adj, self.edhnn_ui_n, self.ui_adj) + res
            else:
                #ego_embeddings = self.lns[0](self.hgcn_layer(self.sparse_norm_adj, ego_embeddings, act=False)) + res
                ego_embeddings = self.lns[k](self.hgcn_layer(self.sparse_norm_adj, ego_embeddings, act=False)) + res
                #ego_embeddings = self.edhnn_layers[k](ego_embeddings, sparse_norm_adj, self.edhnn_ui_n) + res
            all_embeddings += [ego_embeddings]

        user_all_embeddings = all_embeddings[-1][:self.data.n_users]
        item_all_embeddings = all_embeddings[-1][self.data.n_users:]
        return user_all_embeddings, item_all_embeddings
    
    # def forward(self, ego_embeddings, sparse_norm_adj):
    #     res = ego_embeddings
    #     res_user = ego_embeddings[:self.data.n_users]
    #     res_item = ego_embeddings[self.data.n_users:]
    #     all_embeddings = []

    #     for k in range(self.layers):
    #         if k != self.layers - 1: 
    #             user_ego = self.edhnn_layers[k](ego_embeddings[:self.data.n_users], self.hyper_uu, self.edhnn_ui_n, self.ui_adj) + res_user
    #             item_ego = self.edhnn_layers[k](ego_embeddings[self.data.n_users:], self.hyper_ii, self.edhnn_ui_n, self.ui_adj) + res_item
    #             ego_embeddings = torch.cat([user_ego, item_ego], dim=0)
    #         else:
    #             #ego_embeddings = self.lns[0](self.hgcn_layer(self.sparse_norm_adj, ego_embeddings, act=False)) + res
    #             ego_embeddings = self.lns[k](self.hgcn_layer(self.sparse_norm_adj, ego_embeddings, act=False)) + res
    #             #ego_embeddings = self.edhnn_layers[k](ego_embeddings, sparse_norm_adj, self.edhnn_ui_n) + res
    #         all_embeddings += [ego_embeddings]

    #     user_all_embeddings = all_embeddings[-1][:self.data.n_users]
    #     item_all_embeddings = all_embeddings[-1][self.data.n_users:]
    #     return user_all_embeddings, item_all_embeddings


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
        self.hwnn_args = {
            'filters':self.out_channels,
            'dropout': 0.01,
            'ncount': self.ncount,
            'mcount': self.ncount,
            'feature_number': self.out_channels
        }
        self.sparse_norm_adj = self.sparse_norm_adj  = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(device)
        self.wavelet_layers = nn.ModuleList([HWNN(self.hwnn_args['filters'], self.hwnn_args['dropout'], self.hwnn_args['ncount'], self. hwnn_args['mcount'], self.hwnn_args['feature_number'], self.device, self.data) for i in range(self.layers)])
        self.wavelet_layers_uu = nn.ModuleList([HWNN(self.hwnn_args['filters'], self.hwnn_args['dropout'], self.data.n_users, self. hwnn_args['mcount'], self.hwnn_args['feature_number'], self.device, self.data) for i in range(self.layers)])
        self.wavelet_layers_ii = nn.ModuleList([HWNN(self.hwnn_args['filters'], self.hwnn_args['dropout'], self.data.n_items, self. hwnn_args['mcount'], self.hwnn_args['feature_number'], self.device, self.data) for i in range(self.layers)])
        self.hgcn_layers = nn.ModuleList([HGCNConv(leaky=0.5) for i in range(self.layers)])

        self.ui_adj = self.data.ui_adj
        self.hyper_uu = torch.tensor(self.ui_adj.todense()[:self.data.n_users, self.data.n_users:], requires_grad=False).to(device)# sparse_norm_adj: [user+item, user+item]
        self.hyper_ii = torch.tensor(self.ui_adj.todense()[self.data.n_users:, :self.data.n_users], requires_grad=False).to(device)

        self.edhnn_ui_n = self.data.n_users + self.data.n_items

        self.lns = torch.nn.LayerNorm(emb_size)
        

        # self.wavelet = WaveletTransform(self.ui_adj, approx=True)
        # self.wavelet_hypergraph = self.wavelet.generate_hypergraph()

    # def forward(self, ego_embeddings, sparse_norm_adj):
    #     res = ego_embeddings
    #     all_embeddings = []

    #     for k in range(self.layers):
    #         if k != self.layers - 1:
    #             ego_embeddings = self.wavelet_layers[k](ego_embeddings, sparse_norm_adj, '') + res
    #         else:
    #             ego_embeddings = self.wavelet_layers[k](ego_embeddings, sparse_norm_adj, '') + res
    #         all_embeddings += [ego_embeddings]

    #     user_all_embeddings = all_embeddings[-1][:self.data.n_users]
    #     item_all_embeddings = all_embeddings[-1][self.data.n_users:]
    #     return user_all_embeddings, item_all_embeddings

    def forward(self, ego_embeddings, sparse_norm_adj):
        res = ego_embeddings
        res_user = ego_embeddings[:self.data.n_users]
        res_item = ego_embeddings[self.data.n_users:]
        all_embeddings = []

        for k in range(self.layers):
            if k != self.layers - 1: 
                user_ego = self.wavelet_layers_uu[k](ego_embeddings[:self.data.n_users], self.hyper_uu, 'simple') + res_user
                item_ego = self.wavelet_layers_ii[k](ego_embeddings[self.data.n_users:], self.hyper_ii, 'simple') + res_item
                ego_embeddings = torch.cat([user_ego, item_ego], dim=0)
            else:
                #ego_embeddings = self.lns[0](self.hgcn_layer(self.sparse_norm_adj, ego_embeddings, act=False)) + res
                ego_embeddings = self.lns(self.hgcn_layers[0](self.sparse_norm_adj, ego_embeddings, act=False)) + res
                #ego_embeddings = self.edhnn_layers[k](ego_embeddings, sparse_norm_adj, self.edhnn_ui_n) + res
                # user_ego = self.wavelet_layers_uu[k](ego_embeddings[:self.data.n_users], self.hyper_uu, 'simple') + res_user
                # item_ego = self.wavelet_layers_ii[k](ego_embeddings[self.data.n_users:], self.hyper_ii, 'simple') + res_item
                # ego_embeddings = torch.cat([user_ego, item_ego], dim=0)
            all_embeddings += [ego_embeddings]

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

class EquivSetGNN(nn.Module):
    def __init__(self, num_features, args, dense_hypergraph, data, ncount, mcount):
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
        
        self.conv = EquivSetConv(args['MLP_hidden'], args['MLP_hidden'], ncount, mcount, mlp1_layers=self.mlp1_layers, mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers, alpha=args['restart_alpha'], aggr=args['aggregate'],
            dropout=args['dropout'], normalization=args['normalization'], input_norm=args['AllSet_input_norm'], hypergraph=dense_hypergraph, data=self.data)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, sparse_norm_adj, n_nodes, ui_adj, act=True):
        '''my code'''
        # build bipartite graphs for user and item(star expension)
        #V, E = self.generate_V_E(n_nodes, sparse_norm_adj)
        '''---'''
        lamda, alpha = 0.5, 0.1
        x = self.dropout(x)
        x = F.relu(self.lin_in(x))
        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, sparse_norm_adj, x0, ui_adj, act)
            x = self.act(x)
        x = self.dropout(x)
        return x

    '''my_code'''
    def generate_V_E(self, n_nodes, hypergraph):
        
        new_connections = self.build_new_hypergraph(hypergraph)

        hypergraph = hypergraph.to_dense()
        non_zero_indices = torch.nonzero(hypergraph > 0)
        n_connections = non_zero_indices.size(0)
        
        vertex_n = hypergraph.shape[0]
        edge_n = hypergraph.shape[1]

        V = torch.zeros(n_connections, dtype=torch.long)# [120938]
        E = torch.zeros(n_connections, dtype=torch.long)# [120938]
        
        V = non_zero_indices[:, 0].to(device)
        E = non_zero_indices[:, 1].to(device)
        
        return V, E

    def build_new_hypergraph(self, hypergraph):

        hypergraph = hypergraph.to_dense()
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
        
class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, ncount, mcount, mlp1_layers=1, mlp2_layers=1,
        mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False, hypergraph=None, data=None):
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
        self.data = data
        self.device = device
        self.hwnn_args = {
            'filters': out_features,
            'dropout': 0.01,
            'ncount': ncount,
            'mcount': mcount,
            'feature_number': out_features
        }
        #self.hwnn_layers = nn.ModuleList([HWNN(self.hwnn_args['filters'], self.hwnn_args['dropout'], self.hwnn_args['ncount'], self. hwnn_args['mcount'], self.hwnn_args['feature_number'], self.device, self.data) for i in range(2)])
        self.hgcn_layers = nn.ModuleList([HGCNConv(0.5) for i in range(2)])
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
    
    def forward(self, X, sparse_norm_adj, X0, ui_adj, act=True):
        N = X.shape[-2]# 2708(node_num)
        
        Xve = self.W1(X) # [nnz, C]([7494, 256]) # self.W1(X): [node_num, 256]
        
        Xe = self.lns[0](self.hgcn_layers[0](sparse_norm_adj, Xve, act=True)) + Xve
        # edge gathering
        Xev = self.W2(torch.cat([X, Xe], -1))
        Xev = self.mean_pooling(Xev)
        X_v = self.lns[1](self.hgcn_layers[1](sparse_norm_adj, Xev, act=True)) + Xev

        X = X_v
        X = (1-self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X
    
    # def forward(self, X, sparse_norm_adj, X0, ui_adj, act=True):
    #     N = X.shape[-2]# 2708(node_num)
    #     vertex, edges = self.generate_V_E(ui_adj.shape[0], sparse_norm_adj)
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
    
    def generate_V_E(self, n_nodes, hypergraph):
        
        new_connections = self.build_new_hypergraph(hypergraph)

        hypergraph = hypergraph.to_dense()
        non_zero_indices = torch.nonzero(hypergraph > 0)
        n_connections = non_zero_indices.size(0)
        
        vertex_n = hypergraph.shape[0]
        edge_n = hypergraph.shape[1]

        V = torch.zeros(n_connections, dtype=torch.long)# [120938]
        E = torch.zeros(n_connections, dtype=torch.long)# [120938]
        
        V = non_zero_indices[:, 0].to(device)
        E = non_zero_indices[:, 1].to(device)
        
        return V, E

    def build_new_hypergraph(self, hypergraph):

        hypergraph = hypergraph.to_dense()
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
        self.K1 = 1
        self.K2 = 1
        self.ncount = ncount
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
            Theta = wavelet_hypergraph.t()
            Theta_t = torch.transpose(Theta, 0, 1)
        elif msg == "simple_msg_v":
            Theta = wavelet_hypergraph
            Theta_t = torch.transpose(Theta, 0, 1)
        elif msg == 'simple':
            w_hypergraph = wavelet_hypergraph.to(device).to_sparse()
            w_hypergraph_t = torch.transpose(w_hypergraph, 0, 1)
            Theta = torch.sparse.mm(wavelet_hypergraph, w_hypergraph_t)
            Theta_t = torch.transpose(Theta, 0, 1)
        else:
            Theta = wavelet_hypergraph
            Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            eye_ncount = torch.eye(self.ncount, device=device).to_sparse()
            poly = self.par[0] * eye_ncount
            Theta_mul = torch.eye(self.ncount).to(device).to_sparse()
            for ind in range(1, self.K1):
                Theta_mul = torch.sparse.mm(Theta_mul, Theta).to_sparse()
                #poly.add_(self.par[ind] * Theta_mul)
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * eye_ncount
            Theta_mul = eye_ncount
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = torch.sparse.mm(Theta_mul, Theta_t).to_sparse()
                #poly_t.add_(self.par[ind] * Theta_mul)
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
        # Floating Point Operation Per Second
        # caculates the complexity of MLP
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
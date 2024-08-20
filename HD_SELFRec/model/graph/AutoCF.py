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
import torch.nn.functional as F

'''
python main.py --model=AutoCF --dataset=lastfm  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=AutoCF --dataset=ml-1m  --lrate=0.0001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=AutoCF --dataset=amazon_books  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.001 --early_stopping_steps=20 --seed=20
python main.py --model=AutoCF --dataset=steam  --lrate=0.001 --weight_decay=0 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=AutoCF --dataset=yelp  --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
'''
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20\
class AutoCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(AutoCF, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        args = OptionConf(self.config['AutoCF'])
        self.kwargs = kwargs
        self.gt_layers = int(self.config['gt_layer'])
        self.gcn_layers = int(self.config['gcn_layer'])
        self.reg = float(kwargs['reg'])
        self.ssl_reg = float(self.config['ssl_reg'])
        self.lr_decay  = float(kwargs['lr_decay'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.lRate = float(kwargs['lrate'])
        self.wdecay = float(kwargs['weight_decay'])
        self.fix_steps = int(self.config['fix_steps'])
        self.head_num = int(self.config['head_num'])
        self.seed_num = int(self.config['seed_num'])
        self.mask_depth = int(self.config['mask_depth'])
        self.keep_rate = float(self.config['keep_rate'])

        self.model = AutoCF_Encoder(self.data, self.emb_size, self.gt_layers, self.gcn_layers, self.head_num, self.seed_num, self.mask_depth, self.keep_rate)
        self.encoderAdj, self.decoderAdj = None, None

    def train(self, load_pretrained):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.wdecay)

        lst_train_losses = []
        lst_rec_losses = []
        lst_reg_losses = []
        lst_cl_losses = []
        lst_performances = []
        recall_list = []

        for epoch in range(self.maxEpoch):
            train_losses = []
            rec_losses = []
            reg_losses = []
            cl_losses = []
            ep_loss = 0

            s_train = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                if n % self.fix_steps == 0:
                    sampScores, seeds = model.sample_subgraphs()
                    encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model(encoderAdj, decoderAdj)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = (-torch.sum(user_emb * pos_item_emb, dim=-1)).mean()
                reg_loss =  l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb) /self.batch_size
                cl_loss = (self.contrast(user_idx, rec_user_emb) + self.contrast(pos_idx, rec_item_emb)) * self.ssl_reg + self.contrast(user_idx, rec_user_emb, rec_item_emb)
                batch_loss = rec_loss + reg_loss + cl_loss

                train_losses.append(batch_loss.item())
                rec_losses.append(rec_loss.item())
                reg_losses.append(reg_loss.item())
                cl_losses.append(cl_loss.item())
    
                if n % self.fix_steps == 0:
                    localGlobalLoss = -sampScores.mean()
                    batch_loss += localGlobalLoss
                
                ep_loss += batch_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            e_train = time.time() 
            tr_time = e_train - s_train 

            train_loss = np.mean(train_losses)
            rec_loss = np.mean(rec_losses)
            reg_loss = np.mean(reg_losses)
            cl_loss = np.mean(cl_losses)

            lst_train_losses.append([epoch, train_loss])
            lst_rec_losses.append([epoch,rec_loss])
            lst_reg_losses.append([epoch, reg_loss])
            lst_cl_losses.append([epoch, cl_loss])

            with torch.no_grad():
                # make ui adj
                self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()

                self.user_emb, self.item_emb = self.model.forward(self.sparse_norm_adj, self.sparse_norm_adj)

                cur_data, data_ep = self.fast_evaluation(epoch, train_time=tr_time)
                lst_performances.append(data_ep)
                
                cur_recall =  float(cur_data[2].split(':')[1])
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                if should_stop:
                    break 
        
        print(lst_train_losses)
        self.save_loss(lst_train_losses, lst_rec_losses, lst_reg_losses)
        self.save_perfomance_training(lst_performances)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    
    def contrast(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            scores = torch.log(torch.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
        else:
            uniqNodes = torch.unique(nodes)
            pckEmbeds = allEmbeds[uniqNodes]
            scores = torch.log(torch.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
        return scores
    
    def save(self):
        with torch.no_grad():
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()
            self.best_user_emb, self.best_item_emb = self.model.forward(self.sparse_norm_adj, self.sparse_norm_adj)
            print("Saving")
            self.save_model(self.model)     

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class AutoCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, gt_layers, gcn_layers, head_num, seed_num, mask_depth, keep_rate):
        super(AutoCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.head_num = head_num
        self.gt_Layers = gt_layers
        self.gcn_Layers = gcn_layers
        self.seed_num = seed_num
        self.mask_depth = mask_depth
        self.keep_rate = keep_rate
        self.user_num = self.data.n_users
        self.item_num = self.data.n_items

        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.all_one_adj = self.make_all_one_adj()

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gcn_Layers)])
        self.gtLayers = nn.Sequential(*[GTLayer(self.head_num, self.latent_size) for i in range(self.gt_Layers)])

        self.masker = RandomMaskSubgraphs(self.mask_depth, self.keep_rate, self.user_num, self.item_num)
        self.sampler = LocalGraph(self.seed_num)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.latent_size)))
        })
        return embedding_dict

    def make_all_one_adj(self):
        idxs = self.sparse_norm_adj._indices()
        vals = torch.ones_like(self.sparse_norm_adj._values())
        shape = self.norm_adj.shape
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
    
    def get_ego_embeds(self):
        return torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
    
    def sample_subgraphs(self):
        return self.sampler(self.all_one_adj, self.get_ego_embeds())
    
    def mask_subgraphs(self, seeds):
        return self.masker(self.sparse_norm_adj, seeds)
    
    def forward(self, encoder_adj, decoder_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k, gcn in enumerate(self.gcnLayers):
            ego_embeddings = gcn(encoder_adj, all_embeddings[-1])
            all_embeddings.append(ego_embeddings)
        if decoder_adj is not None:
            for gt in self.gtLayers:
                ego_embeddings = gt(decoder_adj, all_embeddings[-1])
                all_embeddings.append(ego_embeddings)
        all_embeddings = sum(all_embeddings)

        return ego_embeddings[:self.user_num], ego_embeddings[self.user_num:]
    
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

class GTLayer(nn.Module):
    def __init__(self, head_num, emb_size):
        super(GTLayer, self).__init__()

        self.head_num = head_num
        self.embedding_size = emb_size
        init = nn.init.xavier_uniform_

        self.qTrans = nn.Parameter(init(torch.empty(self.embedding_size, self.embedding_size)))
        self.kTrans = nn.Parameter(init(torch.empty(self.embedding_size, self.embedding_size)))
        self.vTrans = nn.Parameter(init(torch.empty(self.embedding_size, self.embedding_size)))
    
    def forward(self, adj, embeds):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        
        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        tem = torch.zeros([adj.shape[0], self.head_num]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8) # eh
        
        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.embedding_size])
        tem = torch.zeros([adj.shape[0], self.embedding_size]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
        return resEmbeds
    
class LocalGraph(nn.Module):
    def __init__(self, seed_num):
        super(LocalGraph, self).__init__()
        self.seed_num = seed_num
    
    def makeNoise(self, scores):
        noise = torch.rand(scores.shape).cuda()
        noise[noise == 0] = 1e-8
        noise = -torch.log(-torch.log(noise))
        return torch.log(scores) + noise
    
    def forward(self, allOneAdj, embeds):
        # allOneAdj should be without self-loop
        # embeds should be zero-order embeds
        order = torch.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = torch.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (torch.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (torch.spmm(allOneAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        scores = torch.sigmoid(torch.sum(subgraphEmbeds * embeds, dim=-1))
        scores = self.makeNoise(scores)
        _, seeds = torch.topk(scores, self.seed_num)
        return scores, seeds
    
class RandomMaskSubgraphs(nn.Module):
    def __init__(self, mask_depth, keep_rate, user_num, item_num):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.mask_depth = mask_depth
        self.keep_rate = keep_rate
        self.user_num = user_num
        self.item_num = item_num
    
    def normalizeAdj(self, adj):
        degree = torch.pow(torch.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return torch.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(self.mask_depth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = torch.logical_or(rowIdct, colIdct)

                if i != self.mask_depth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[torch.logical_not(idct)]
                cols = cols[torch.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = torch.unique(torch.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)
        sampNum = int((self.user_num + self.item_num) * self.keep_rate)
        sampedNodes = torch.randint(self.user_num + self.item_num, size=[sampNum]).cuda()
        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = torch.unique(torch.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (self.user_num + self.item_num)), tem.shape[0], (self.user_num + self.item_num))
        maskNodes.append(sampedNodes)
        maskNodes = torch.unique(torch.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (self.user_num + self.item_num)), maskNodes.shape[0], (self.user_num + self.item_num))
            self.flag = True
            print('-----')

        
        encoder_adj = self.normalizeAdj(torch.sparse.FloatTensor(torch.stack([rows, cols], dim=0), torch.ones_like(rows).cuda(), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[torch.randint(temNum, size=[adj._values().shape[0]]).cuda()]
        temCols = maskNodes[torch.randint(temNum, size=[adj._values().shape[0]]).cuda()]

        newRows = torch.concat([temRows, temCols, torch.arange(self.user_num+self.item_num).cuda(), rows])
        newCols = torch.concat([temCols, temRows, torch.arange(self.user_num+self.item_num).cuda(), cols])

        # filter duplicated
        hashVal = newRows * (self.user_num + self.item_num) + newCols
        hashVal = torch.unique(hashVal)
        newCols = hashVal % (self.user_num + self.item_num)
        newRows = ((hashVal - newCols) / (self.user_num + self.item_num)).long()


        decoder_adj = torch.sparse.FloatTensor(torch.stack([newRows, newCols], dim=0), torch.ones_like(newRows).cuda().float(), adj.shape)
        return encoder_adj, decoder_adj
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
import pickle

'''
python main.py --model=LRMRec --dataset=lastfm  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=LRMRec --dataset=amazon_books  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.001 --early_stopping_steps=20 --seed=20
python main.py --model=LRMRec --dataset=steam  --lrate=0.01 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
python main.py --model=LRMRec --dataset=yelp  --lrate=0.001 --weight_decay=5e-6 --drop_rate=0.2 --p=0.3 --cl_rate=1e-05 --temp=0.2 --reg=0.1 --early_stopping_steps=20 --seed=20
'''
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20\
class LRMRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(LRMRec, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
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
        self.mask_ratio = float(self.config['mask_ratio'])
        self.recon_weight = float(self.config['recon_weight'])
        self.re_temperature = float(self.config['retemperature'])

        usrprf_embeds_path = "./dataset/{}/usr_emb_np.pkl".format(self.config['dataset'])
        itmprf_embeds_path = "./dataset/{}/itm_emb_np.pkl".format(self.config['dataset'])

        with open(usrprf_embeds_path, 'rb') as f:
            usrprf_embeds = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            itmprf_embeds = pickle.load(f)

        self.usrprf_embeds = usrprf_embeds
        self.itmprf_embeds = itmprf_embeds

        self.model = LRMRec_Encoder(self.data, self.emb_size, self.gt_layers, self.gcn_layers, self.head_num, self.seed_num, self.mask_depth, self.keep_rate, self.mask_ratio, self.recon_weight, self.re_temperature, self.usrprf_embeds, itmprf_embeds)
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

                # cal_loss
                masked_user_embeds, masked_item_embeds, seeds = model._mask()
                rec_user_emb, rec_item_emb = model.forward(encoderAdj, decoderAdj, masked_user_embeds, masked_item_embeds)
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = (-torch.sum(user_emb * pos_item_emb, dim=-1)).mean()
                reg_loss =  l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb) /self.batch_size
                cl_loss = (self.contrast(user_idx, rec_user_emb) + self.contrast(pos_idx, rec_item_emb)) * self.ssl_reg + self.contrast(user_idx, rec_user_emb, rec_item_emb)
                recon_loss = self.recon_weight * model._reconstruction(torch.concat([rec_user_emb, rec_item_emb], axis=0), seeds)
               
                batch_loss = rec_loss + reg_loss + cl_loss + recon_loss

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

class LRMRec_Encoder(nn.Module):
    def __init__(self, data, emb_size, gt_layers, gcn_layers, head_num, seed_num, mask_depth, keep_rate, mask_ratio, recon_weight, re_temperature, usrprf_embeds, itmprf_embeds):
        super(LRMRec_Encoder, self).__init__()
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
        self.init = nn.init.xavier_uniform_
        self.mask_ratio = mask_ratio
        self.recon_weight = recon_weight
        self.re_temperature = re_temperature

        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.all_one_adj = self.make_all_one_adj()

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gcn_Layers)])
        self.gtLayers = nn.Sequential(*[GTLayer(self.head_num, self.latent_size) for i in range(self.gt_Layers)])

        self.masker = RandomMaskSubgraphs(self.mask_depth, self.keep_rate, self.user_num, self.item_num)
        self.sampler = LocalGraph(self.seed_num)

        # get semantic embeddings learned from profiles
        usrprf_embeds = torch.tensor(usrprf_embeds).float().cuda()
        itmprf_embeds = torch.tensor(itmprf_embeds).float().cuda()
        self.prf_embeds = torch.concat([usrprf_embeds, itmprf_embeds], dim=0)

        # generative process
        self.gene_masker = NodeMask(self.mask_ratio, self.latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_size, (self.prf_embeds.shape[1] + self.latent_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.prf_embeds.shape[1] + self.latent_size) // 2, self.prf_embeds.shape[1])
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                self.init(m.weight)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.latent_size)))
        })
        return embedding_dict

    def _mask(self):
        embeds = torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        masked_embeds, seeds = self.gene_masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds

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
    
    def ssl_con_loss(self, x, y, temp):
        x = F.normalize(x)
        y = F.normalize(y)
        mole = torch.exp(torch.sum(x * y, dim=1) / temp)
        deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)

        return -torch.log(mole / (deno + 1e-8) + 1e-8).mean()

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = self.ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def forward(self, encoder_adj, decoder_adj=None, masked_user_embeds=None, masked_item_embeds=None):
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        else:
            embeds = torch.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embedsLst = [embeds]
        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoder_adj, embedsLst[-1])
            embedsLst.append(embeds)
        if decoder_adj is not None:
            for gt in self.gtLayers:
                embeds = gt(decoder_adj, embedsLst[-1])
                embedsLst.append(embeds)
        embeds = sum(embedsLst)

        return embeds[:self.user_num], embeds[self.user_num:]
    
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

class NodeMask(nn.Module):
    """ Mask nodes with learnable tokens
    """
    def __init__(self, mask_ratio, embedding_size):
        super(NodeMask, self).__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, embedding_size))
    
    def forward(self, embeds):
        seeds = np.random.choice(embeds.shape[0], size=max(int(embeds.shape[0] * self.mask_ratio), 1), replace=False)
        seeds = torch.LongTensor(seeds).cuda()
        mask = torch.ones(embeds.shape[0]).cuda()
        mask[seeds] = 0
        mask = mask.view(-1, 1)
        masked_embeds = embeds * mask + self.mask_token * (1. - mask)
        return masked_embeds, seeds 
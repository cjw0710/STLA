import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import torch
from dataLoader import dataProcess
from module import SocialAN, Fusion, InteractionAN,LightGCN
from torch.nn.parameter import Parameter
from torch_scatter import scatter

from torch_geometric.nn import GCNConv

from utils.Setup import trans_to_cuda

class DeDiff(nn.Module):
    def __init__(self, args):
        super(DeDiff, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num
        self.dim = args.dim  
        self.GCN1 = LightGCN(args)
        self.GCN2 = LightGCN(args)
        self.GCN3 = LightGCN(args)
        self.GCN4 = LightGCN(args)
        self.SSAN = SocialAN(args)
        self.TEAN = InteractionAN(args)
        self.Fusion = Fusion(args)
        self.Fusion2 = Fusion(args)
        self.Predict = nn.Linear(self.dim, self.user_num)
        self.Debasing= Parameter(torch.FloatTensor(self.user_num,self.user_num))

        self.mlp_t = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )

        self.mlp_s = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.reset_parameters()
        self.UEm = nn.Embedding(self.user_num, self.dim, padding_idx=0)
    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, args, data,info):
        cascade, cas_mask, label, label_mask, neighbor, relation, dis,timestamp = dataProcess(args, data)
        process_data = {}
        interaction_graph = trans_to_cuda(info['A_interaction'])
        social_graph =trans_to_cuda(info['A_social'])
        D = self.Debasing
        casual_interaction_graph = torch.einsum('nm,mk->nk', interaction_graph, D)
        casual_social_graph = torch.einsum('nm,mk->nk', social_graph, D)
        bias_interaction_graph = interaction_graph - casual_interaction_graph
        bias_social_graph = social_graph - casual_social_graph

        h_casual_interaction = self.GCN1(self.UEm.weight, casual_interaction_graph)
        h_bias_interaction = self.GCN2(self.UEm.weight, bias_interaction_graph)
        h_casual_social = self.GCN3(self.UEm.weight, casual_social_graph)
        h_bias_social = self.GCN4(self.UEm.weight, bias_social_graph)

        process_data['embedding_isc'] = h_casual_interaction + h_casual_social
        process_data['embedding_isb'] = h_bias_interaction + h_bias_social

        embedding_tc_proxy = h_casual_interaction.mean(dim=0)
        embedding_sc_proxy_sum = torch.einsum('ne, nm->me', h_casual_social, social_graph)
        embedding_sc_proxy = embedding_sc_proxy_sum.mean(dim=0)

        e_T_prime = self.mlp_t(h_casual_interaction)
        e_S_prime = self.mlp_s(h_casual_social)
        p_T_prime = self.mlp_t(embedding_tc_proxy)
        p_S_prime = self.mlp_s(embedding_sc_proxy)

        process_data['e_T_prime'] = e_T_prime
        process_data['e_S_prime'] = e_S_prime
        process_data['p_T_prime'] = p_T_prime
        process_data['p_S_prime'] = p_S_prime

        user_embed1 = F.embedding(cascade, h_casual_interaction, padding_idx=0)
        user_embed2 = F.embedding(cascade, h_casual_social, padding_idx=0)

        casEmbed = self.Fusion2(user_embed1, user_embed2)
        ht = self.TEAN(casEmbed, cas_mask, timestamp)
        hs = self.SSAN(casEmbed, cas_mask, dis)
        h = self.Fusion(ht, hs)

        user_embed3 = F.embedding(cascade, h_bias_interaction, padding_idx=0)
        user_embed4 = F.embedding(cascade, h_bias_social, padding_idx=0)

        casEmbed = self.Fusion2(user_embed3, user_embed4)
        ht = self.TEAN(casEmbed, cas_mask, timestamp)
        hs = self.SSAN(casEmbed, cas_mask, dis)
        h_b = self.Fusion(ht, hs)
        process_data['h_b'] = h_b
        process_data['frequency'] = info['frequency']  
        pred_user = self.Predict(h) + label_mask
        pred_user = pred_user.view(-1, pred_user.size(-1))

        return pred_user, label, process_data

def loss_function(pred, label, process_state = None):
    
    loss_function_rec = nn.CrossEntropyLoss(ignore_index=0)  
    loss_rec = loss_function_rec(pred, label)
    loss = loss_rec
    if process_state is not None:
        e_T_prime = process_state['e_T_prime']
        e_S_prime = process_state['e_S_prime']
        p_T_prime = process_state['p_T_prime']
        p_S_prime = process_state['p_S_prime']
        pos_score_T = torch.matmul(e_T_prime, p_T_prime.T)  
        neg_score_T = torch.matmul(e_T_prime, p_S_prime.T)  
        pos_score_G = torch.matmul(e_S_prime, p_S_prime.T)  
        neg_score_G = torch.matmul(e_S_prime, p_T_prime.T)  

        loss_ts = -torch.log(torch.sigmoid(pos_score_T - neg_score_T)).sum()  
        loss_ts += -torch.log(torch.sigmoid(pos_score_G - neg_score_G)).sum()  
        loss_ts /= e_T_prime.size(0)
        loss+= loss_ts

        embedding_isc = process_state['embedding_isc']
        embedding_isb = process_state['embedding_isb']
        z_c = embedding_isc.mean(dim=0)  
        z_b = embedding_isb.mean(dim=0)  

        dist_c_to_c = torch.norm(embedding_isc - z_c, p=2, dim=1).pow(2).sum()  
        dist_c_to_b = torch.norm(embedding_isc - z_b, p=2, dim=1).pow(2).sum()  

        m = 1e-5  
        loss_cb = (dist_c_to_c - dist_c_to_b + m) / e_T_prime.size(0)  
        loss += loss_cb

        frequency = process_state['frequency']
        h_b = process_state['h_b']
        epsi = F.softmax(frequency, dim=-1)
        matching_scores_bias = torch.einsum('bse,ne->bsn', h_b, embedding_isb)  
        predicted_bias = F.softmax(matching_scores_bias, dim=-1).mean(dim=1)  
        loss_bias = F.kl_div(predicted_bias.log(), epsi, reduction='sum') 
        loss = loss + loss_bias
    return loss
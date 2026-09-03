import json
import numpy as np
from scipy.sparse import load_npz

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
from utils.Setup import trans_to_cuda

def create_dataloaders(args):
    train_dataset = CascadeData(args, args.cascade_train_path)
    val_dataset = CascadeData(args, args.cascade_valid_path)
    test_dataset = CascadeData(args, args.cascade_test_path)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader,train_dataset

class CascadeData(Dataset):
    
    def __init__(self, args, dataPath):
        
        self.max_len = args.max_len  
        self.EOS = args.user_num - 1  
        self.sample_k = args.sample_k  
        self.seed = args.seed  
        with open(dataPath, 'r') as cas_file:
            self.cascade_data = json.load(cas_file)
        self.G = load_npz(args.graphPath)
        self.sampler = Sample4Cas(G=self.G, sample_k=self.sample_k, random_seed=self.seed)
        dis_mat = np.load(args.dis_path, allow_pickle=True)
        self.dis_matrix = torch.tensor(dis_mat)

    def get_dis(self, seq_list):
        indices = torch.tensor(seq_list)
        dis_Mat = torch.index_select(self.dis_matrix, 0, indices)
        dis_Mat = torch.index_select(dis_Mat, 1, indices)
        size_k = self.max_len
        dis_matrix = torch.full((size_k, size_k), self.EOS)

        l = dis_Mat.size(0)
        dis_matrix[:l, :l] = dis_Mat
        return dis_matrix

    def __len__(self) -> int:
        
        return len(self.cascade_data)

    def __getitem__(self, idx: int) -> dict:
        cascade = self.cascade_data[idx]['cascade'] + [self.EOS]
        timestamp = self.cascade_data[idx]['timestamp']  
        cas_raw = cascade[:self.max_len + 1] if len(cascade) > self.max_len + 1 else cascade
        timestamp_raw = timestamp[:self.max_len + 1] if len(timestamp) > self.max_len + 1 else timestamp
        dis_m = self.get_dis(cas_raw[:-1])
        neigh_raw, rela_raw = self.sampler.sample_cas(cas_raw[:-1])
        cas_raw = torch.Tensor(cas_raw).long().squeeze()
        timestamp_raw = torch.Tensor(timestamp_raw).long().squeeze()
        cascade = pad_1d_tensor(cas_raw, self.max_len + 1)
        neighbor = pad_2d_tensor(neigh_raw, self.max_len)
        relation = pad_2d_tensor(rela_raw, self.max_len)
        timestamp = pad_1d_tensor(timestamp_raw, self.max_len+1)
        data = dict(
            cascade=cascade,
            timestamp=timestamp,
            neighbor=neighbor,
            relation=relation,
            dis_matrix=dis_m.squeeze()
        )
        return data

def dataProcess(args, data):
    cas_pad = data['cascade']
    cascade = trans_to_cuda(cas_pad[:, :-1])
    cas_mask = trans_to_cuda((cascade == 0))
    label = trans_to_cuda(cas_pad[:, 1:]).contiguous().view(-1)
    label_mask = trans_to_cuda(get_previous_user_mask(cascade, args.user_num))
    neighbor = trans_to_cuda(data['neighbor'])
    relation = trans_to_cuda(data['relation'])
    dis = trans_to_cuda(data['dis_matrix'])

    ts = data['timestamp'][:, :-1]
    ts = trans_to_cuda(ts)
    min_ts = ts.min(dim=1, keepdim=True).values
    max_ts = ts.max(dim=1, keepdim=True).values
    span = (max_ts - min_ts).clamp(min=1.0)
    bin_size = span / args.time_intervals
    new_ts = ((ts - min_ts) / bin_size).floor().long().clamp(min=0, max=args.time_intervals-1)

    return cascade, cas_mask, label, label_mask, neighbor, relation, dis, new_ts

class Sample4Cas:
    
    def __init__(self, G, sample_k, random_seed=None):
        
        self.G = G  
        self.sample_k = sample_k  
        self.random_seed = random_seed  
        self.rng = np.random.default_rng(seed=random_seed)  

    def get_neighbors_and_values(self, node_id):
        if node_id < 0 or node_id >= self.G.shape[0]:
            raise ValueError("Invalid node ID.")
        start_idx = self.G.indptr[node_id]  
        end_idx = self.G.indptr[node_id + 1]  

        neighbors = self.G.indices[start_idx:end_idx]  
        values = self.G.data[start_idx:end_idx]  
        if self.random_seed is not None:
            indices = self.rng.permutation(len(neighbors))
            neighbors = neighbors[indices]
            values = values[indices]

        return neighbors, values

    def sample_cas(self, cascade):
        
        cas_len = len(cascade)
        neighbor_ids = torch.zeros((cas_len, self.sample_k), dtype=torch.long)
        neighbor_values = torch.zeros((cas_len, self.sample_k), dtype=torch.float)
        for i, userid in enumerate(cascade):
            neighbors, values = self.get_neighbors_and_values(userid)

            num_neighbors = len(neighbors)
            if num_neighbors < self.sample_k:
                sampled_indices = self.rng.choice(num_neighbors, size=self.sample_k, replace=True)
            else:
                sampled_indices = self.rng.choice(num_neighbors, size=self.sample_k, replace=False)
            neighbor_ids[i] = torch.tensor(neighbors[sampled_indices])
            neighbor_values[i] = torch.tensor(values[sampled_indices])

        return neighbor_ids, neighbor_values

def get_previous_user_mask(seq, user_size):
    
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq

def relation_encoder(rela_num):
    
    if rela_num == 2:
        weight = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [1, 1]],
            device='cuda', requires_grad=False)
    elif rela_num == 3:
        weight = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            device='cuda', requires_grad=False)
    else:
        weight = None
        print('ValueError: Needs further modifications')
    return weight

def pad_1d_tensor(tensor, max_len, pad_value=0):
    
    len_seq = len(tensor)

    if len_seq < max_len:
        pad_len = max_len - len_seq
        padded_tensor = F.pad(tensor, (0, pad_len), value=pad_value)
    else:
        padded_tensor = tensor[:max_len]
    return padded_tensor

def pad_2d_tensor(tensor, max_len, pad_value=0):
    
    tensor = tensor.long().squeeze()
    len_seq, dim = tensor.size()

    if len_seq < max_len:
        pad_len = max_len - len_seq
        padded_tensor = F.pad(tensor, (0, 0, 0, pad_len), value=pad_value)
    else:
        padded_tensor = tensor[:max_len, :]
    return padded_tensor

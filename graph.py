from collections import defaultdict
from scipy.sparse import load_npz
import numpy as np
import torch
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import to_networkx, add_self_loops,from_scipy_sparse_matrix,softmax
from utils.Setup import trans_to_cuda
import os

def construct_graph(args, train_data):
    num = len(train_data)
    k_hop = args.sample_hop
    seq_len = args.max_len
    edges = []
    for i in range(num):
        user = [train_data[i]['cascade']]
        j = k_hop
        while user[0][j] != 0:
            for k in range(k_hop):
                edges.append((user[0][j - k - 1], user[0][j]))
                edges.append((user[0][j], user[0][j - k - 1]))
            if j < seq_len - 1:
                j += 1
            else:
                break
    partner = list(set(edges))
    graph = build_two_graphs(args, partner).cuda()
    return graph


def build_two_graphs(args, co_graph):
    data_name = args.dataset
    friend_path = args.social_graph_path
    EOS = args.user_num - 1
    if os.path.exists(friend_path):
        social_graph_matrix = load_npz(args.social_graph_path)
        friend_out = []
        edge_index, edge_attr = from_scipy_sparse_matrix(social_graph_matrix)
        friend_in = edge_index.t().tolist()
    self_loop = [(i, i) for i in range(args.user_num)]
    if isinstance(co_graph, torch.Tensor):
        edges_partner = co_graph.t().tolist()
    else:
        edges_partner = co_graph
    edges_partner += self_loop
    edges_partner_tensor = torch.LongTensor(edges_partner).t()
    m = to_scipy_sparse_matrix(edges_partner_tensor)
    partner_matrix = Coo2Tensor(m)

    friend_in += [(EOS, EOS)]
    friend_in_tensor = torch.LongTensor(friend_in).t()
    m1 = to_scipy_sparse_matrix(friend_in_tensor)
    friend_in_matrix = Coo2Tensor(m1)

    if len(friend_out) > 0:
        friend_out += [(EOS, EOS)]
        friend_out_tensor = torch.LongTensor(friend_out).t()
        m2 = to_scipy_sparse_matrix(friend_out_tensor)
        friend_out_matrix = Coo2Tensor(m2)
        A_t = torch.stack([friend_in_matrix, friend_out_matrix, partner_matrix], dim=2).to_dense()
    else:
        A_t = torch.stack([friend_in_matrix, partner_matrix], dim=2).to_dense()
    return A_t


def Coo2Tensor(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def get_info(args, train_dataloader, train_dataset):
    info = {}
    user_num = args.user_num
    info['user_num'] = args.user_num
    if os.path.exists(args.frequency_path):
        frequency = torch.load(args.frequency_path)
    else:
        all_data = []
        for batch in train_dataloader:
            all_data.append(batch)
        all_cascades = []
        for batch in train_dataloader:
            batch_size = batch['cascade'].size(0)
            for i in range(batch_size):
                sample = {k: v[i] for k, v in batch.items()}
                all_cascades.append(sample['cascade'])
        counts = defaultdict(int)
        sum = 0
        for cascade in all_cascades:
            for user in cascade:
                counts[user] += 1
                sum += 1
        for user_index, count in counts.items():
            sum += count
        frequency = torch.zeros(user_num)
        for user_index, count in counts.items():
            frequency[user_index] = 1.0 * count / sum
        torch.save(frequency, args.frequency_path)
    info['frequency'] = trans_to_cuda(frequency)

    A_t = construct_graph(args, train_dataset)
    A_interaction = A_t[:, :, 0]
    A_social = A_t[:, :, 1]
    info['A_interaction'] = A_interaction
    info['A_social'] = A_social

    return info

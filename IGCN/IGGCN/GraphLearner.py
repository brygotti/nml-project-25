import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from EEGGraphAttentionLayer import *
from utils import to_cuda


"""
def compute_normalized_laplacian(adj):
    print(f"adj_shape in normalized lapalcian : {adj.shape}")
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm
"""
def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, dim=-1)  # [B, C]
    d_inv_sqrt = torch.pow(rowsum, -0.5)  # [B, C]
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  

    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    L_norm = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)  # [B, C, C]

    return L_norm



class GraphLearner(nn.Module):
    
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=64, metric_type='EEGGraph_attention', alpha=0.2, device=None):
        super(GraphLearner, self).__init__()
        self.device=device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type
        #self.dropout = config['dropout']
        self.num_pers = num_pers
        self.alpha = alpha

       

            
        self.att = EEGGraphAttentionLayer(input_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU(0.2)

        print('[ EEGGraph_attention GraphLearner]')

        

    def forward(self, context, adj, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
       
        attention_head = []
        attention = []
        for _ in range(self.num_pers):
            for i in range(context.size(0)):
                h = to_cuda(context[i])
                ad =to_cuda(adj[i])
                attention_ = self.att(h, ad)
                attention_head.append(attention_)
            attention_head = torch.stack(attention_head, 0)
            attention.append(attention_head)
            attention_head = []

        attention = torch.mean(torch.stack(attention, 0), 0).to(self.device)
        markoff_value = -1e20

        """
        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)
        

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        
        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)
        """

        attention = F.softplus(attention)

        return attention
    """
    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix
    
    
    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists
    """

"""
def get_binarized_kneighbors_graph(features, topk, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    topk = min(topk, attention.size(-1))
    _, knn_ind = torch.topk(attention, topk, dim=-1)
    adj = to_cuda(torch.zeros_like(attention).scatter_(-1, knn_ind, 1), device)
    return adj
"""
import torch
from utils import to_cuda
import numpy as np
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))



def graph_regularization_loss(A_m, A_m_upd, X, A_1, delta, device):
        
        # Graph regularization
        smoothness_ratio = 0.5
        degree_ratio = 0.01
        sparsity_ratio = 0.3
        graph_loss = 0
        for i in range(A_m_upd.shape[0]):
            L = torch.diagflat(torch.sum(A_m_upd[i], -1)) - A_m_upd[i]
            graph_loss += smoothness_ratio * torch.trace(torch.mm(X[i].transpose(-1, -2), torch.mm(L, X[i]))) / int(np.prod(A_m_upd.shape))

        ones_vec = to_cuda(torch.ones(A_m_upd.shape[:-1]), device)
        graph_loss += -degree_ratio * torch.matmul(ones_vec.unsqueeze(1), torch.log(torch.matmul(A_m_upd, ones_vec.unsqueeze(-1)) + 1e-12)).sum() / A_m_upd.shape[0] / A_m_upd.shape[-1]
        graph_loss += sparsity_ratio * torch.sum(torch.pow(A_m_upd, 2)) / int(np.prod(A_m_upd.shape))

        stop_cond =  SquaredFrobeniusNorm(A_m_upd - A_m) / SquaredFrobeniusNorm(A_1) < delta
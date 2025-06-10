import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_maxpool(node_vec, node_mask=None):
    # Maxpool
    # Shape: (batch_size, hidden_size, num_nodes)
    graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
    return graph_embedding

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.state_dim = state_dim

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, x, node_anchor_adj):
        a_in = torch.matmul(node_anchor_adj, x)
        a_out = torch.matmul(node_anchor_adj.transpose(1, 2), x)
        a = torch.cat((a_in, a_out, x), dim=2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * x), dim=2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * x + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_edge_types=1, n_steps=3, nclass=1,device=None):
        super(GGNN, self).__init__()
        self.device=device
        self.state_dim = state_dim
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps
        self.nclass = nclass

        self.propagator = Propogator(self.state_dim)
        self.linear_out = nn.Linear(self.state_dim, 1, bias=False)


        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, node_anchor_adj):
        for i_step in range(self.n_steps):
            x = self.propagator(x, node_anchor_adj)

        H_m = x

        graph_repr = graph_maxpool(H_m.transpose(1, 2))  # Now shape [B, D]
        logits = self.linear_out(graph_repr)



        return H_m, logits
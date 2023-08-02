import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_scatter import scatter
import torch_geometric
import torch.nn.functional as F




class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(in_channels, out_channels * self.heads)
        self.lin_r = nn.Linear(in_channels, out_channels * self.heads)

        self.att_l = nn.Parameter(torch.rand(self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.rand(self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels

        # 1. First apply linear transformation to node embeddings, and split that 
        #    into multiple heads. We use the same representations for source and
        #    target nodes, but apply different linear weights (W_l and W_r)

        h_l = self.lin_l(x).reshape(-1, H, C)
        h_r = self.lin_r(x).reshape(-1, H, C)

        # 2. Calculate alpha vectors for central nodes (alpha_l) and neighbor nodes (alpha_r).
        alpha_l = self.att_l * h_l
        alpha_r = self.att_r * h_r

        # 3. Call propagate function to conduct the message passing.
        out = self.propagate(edge_index=edge_index, x=(h_l, h_r), alpha=(alpha_l, alpha_r), size=size)

        # 4. Transform the output back to the shape of [N, H * C].
        out = out.reshape(-1, H*C)
        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        # 1. Calculate the final attention weights using alpha_i and alpha_j,
        #    and apply leaky Relu.
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)

        # 2. Calculate softmax over the neighbor nodes for all the nodes. Use 
        #    torch_geometric.utils.softmax instead of the one in Pytorch.
        if ptr:
            att_weight = F.softmax(alpha, ptr)
        else:
            att_weight = torch_geometric.utils.softmax(alpha, index)

        # 3. Apply dropout to attention weights (alpha).
        att_weight = F.dropout(att_weight, p=self.dropout)

        # 4. Multiply embeddings and attention weights. As a sanity check, the output
        #    should be of shape [E, H, C].
        out = att_weight * x_j

        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, reduce="sum")
        return out

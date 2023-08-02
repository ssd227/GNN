import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GCN, self).__init__()

        self.num_layers = num_layers

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] +
            [GCNConv(hidden_dim, output_dim)]
        )

        # A list of 1D batch normalization layers
        self.bns = nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

        # Probability of an element getting zeroed
        self.dropout = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax()

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i in range(0, self.num_layers - 1):
            x = self.dropout(self.relu(self.bns[i](
                self.convs[i](x, adj_t))))
        x = self.convs[-1](x, adj_t)

        if self.return_embeds:
            return x
        return self.softmax(x)

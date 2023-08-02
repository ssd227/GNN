import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing


class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True,
                 bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        h1 = self.propagate(edge_index=edge_index, x=(x, x))
        out = self.lin_l(x) + self.lin_r(h1)
        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)
        return out

    def message(self, x_j):
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, reduce="mean")
        return out

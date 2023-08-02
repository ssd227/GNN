import torch
import torch_geometric.nn as pyg_nn
from torch_sparse import SparseTensor, matmul


class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_dst = torch.nn.Linear(in_channels_dst, out_channels)
        self.lin_src = torch.nn.Linear(in_channels_src, out_channels)
        self.lin_update = torch.nn.Linear(out_channels*2, out_channels)

    def forward(self, node_feature_src, node_feature_dst,
        edge_index, size=None):
        return self.propagate(edge_index, node_feature_src=node_feature_src,
                              node_feature_dst=node_feature_dst, size=size)

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = matmul(edge_index, node_feature_src, reduce='mean')
        return out

    def update(self, aggr_out, node_feature_dst):
        dst_out = self.lin_dst(node_feature_dst)
        aggr_out = self.lin_src(aggr_out)

        aggr_out = torch.cat([dst_out, aggr_out], -1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out
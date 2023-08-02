import torch
import torch.nn as nn
import torch.nn.functional as F
from hetero_gnn_conv import HeteroGNNConv
from hetero_gnn_wrapper_conv import HeteroGNNWrapperConv
from deepsnap.hetero_gnn import forward_op


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    # TODO: Implement this function that returns a dictionary of `HeteroGNNConv`
    # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    # object and the `conv` is the `HeteroGNNConv`.

    convs = {}

    all_messages_types = hetero_graph.message_types
    for message_type in all_messages_types:
        if first_layer:
            in_channels_src = hetero_graph.num_node_features(message_type[0])
            in_channels_dst = hetero_graph.num_node_features(message_type[2])
        else:
            in_channels_src = hidden_size
            in_channels_dst = hidden_size
        out_channels = hidden_size
        convs[message_type] = conv(in_channels_src, in_channels_dst, out_channels)

    return convs


class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        ############# Your code here #############
        ## (~10 lines of code)
        ## Note:
        ## 1. For self.convs1 and self.convs2, call generate_convs at first and then
        ##    pass the returned dictionary of `HeteroGNNConv` to `HeteroGNNWrapperConv`.
        ## 2. For self.bns, self.relus and self.post_mps, the keys are node_types.
        ##    `deepsnap.hetero_graph.HeteroGraph.node_types` will be helpful.
        ## 3. Initialize all batchnorms to torch.nn.BatchNorm1d(hidden_size, eps=1).
        ## 4. Initialize all relus to nn.LeakyReLU().
        ## 5. For self.post_mps, each value in the ModuleDict is a linear layer
        ##    where the `out_features` is the number of classes for that node type.
        ##    `deepsnap.hetero_graph.HeteroGraph.num_node_labels(node_type)` will be
        ##    useful.

        self.convs1 = HeteroGNNWrapperConv(
            generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True),
            args, self.aggr)
        self.convs2 = HeteroGNNWrapperConv(
            generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False),
            args, self.aggr)

        all_node_types = hetero_graph.node_types
        for node_type in all_node_types:
            self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1.0)
            self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1.0)
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()
            self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))

    def forward(self, node_feature, edge_index):
        # TODO: Implement the forward function. Notice that `node_feature` is
        # a dictionary of tensors where keys are node types and values are
        # corresponding feature tensors. The `edge_index` is a dictionary of
        # tensors where keys are message types and values are corresponding
        # edge index tensors (with respect to each message type).

        x = node_feature

        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        x = forward_op(x, self.post_mps)

        return x

    def loss(self, preds, y, indices):
        loss = 0
        loss_func = F.cross_entropy

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. For each node type in preds, accumulate computed loss to `loss`
        ## 2. Loss need to be computed with respect to the given index
        ## 3. preds is a dictionary of model predictions keyed by node_type.
        ## 4. indeces is a dictionary of labeled supervision nodes keyed
        ##    by node_type

        for node_type in preds:
            idx = indices[node_type]
            loss += loss_func(preds[node_type][idx], y[node_type][idx])

        return loss

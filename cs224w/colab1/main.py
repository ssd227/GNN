import networkx as nx
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.optim import SGD


def graph_to_edge_list(G):
    edge_list = list(G.edges)
    return edge_list


def edge_list_to_tensor(edge_list):
    # tensor should have the shape [2 x len(edge_list)].
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    return torch.t(edge_index)


def get_pos_edges(G):
    pos_edge_list = graph_to_edge_list(G)
    pos_edge_dic = dict()
    for a, b in pos_edge_list:
        pos_edge_dic[(a, b)] = True
    return pos_edge_dic


def is_negative_edge(edgex, pos_edge_dic):
    node_i, node_j = edgex

    judge1 = (node_i == node_j)
    judge2 = (node_i, node_j) in pos_edge_dic
    judge3 = (node_j, node_i) in pos_edge_dic
    if judge1 or judge2 or judge3:
        return False
    return True


def sample_negative_edges(G, num_neg_samples):
    # TODO: Implement the function that returns a list of negative edges.
    # The number of sampled negative edges is num_neg_samples. You do not
    # need to consider the corner case when the number of possible negative edges
    # is less than num_neg_samples. It should be ok as long as your implementation
    # works on the karate club network. In this implementation, self loops should
    # not be considered as either a positive or negative edge. Also, notice that
    # the karate club network is an undirected graph, if (0, 1) is a positive
    # edge, do you think (1, 0) can be a negative one?

    neg_edge_list = []
    pos_edge_dic = get_pos_edges(G)

    for _ in range(num_neg_samples):
        while True:
            idx = random.randint(0, len(G.nodes) - 1)
            idy = random.randint(0, len(G.nodes) - 1)
            node_i = list(G.nodes)[idx]
            node_j = list(G.nodes)[idy]
            this_edge = (node_i, node_j)

            if is_negative_edge(this_edge, pos_edge_dic):
                new_edge = (node_i, node_j)
                neg_edge_list.append(new_edge)
                pos_edge_dic[new_edge] = True
                break

    return neg_edge_list


def create_node_emb(num_node=34, embedding_dim=16):
    # A torch.nn.Embedding layer will be returned. You do not need to change
    # the values of num_node and embedding_dim. The weight matrix of returned
    # layer should be initialized under uniform distribution.

    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    torch.nn.init.uniform_(emb.weight, a=-1.0, b=1.0)
    return emb


def visualize_emb(emb):
    G = nx.karate_club_graph()
    X = emb.weight.data.numpy()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes(data=True):
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])

    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()
    plt.show()


def accuracy(pred, label):
    accu = 0.0
    accu = torch.sum((pred > 0.5) == label) / pred.shape[0]

    return round(accu.item(), 4)


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    epochs = 25
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(emb.parameters(), lr=learning_rate, weight_decay=0.001)

    for i in range(epochs):
        optimizer.zero_grad()

        M = emb(train_edge.t())  # edge_num * 2 * emb_dim
        dot_product_M = torch.sum(M[:, 0, :] * M[:, 1, :], dim=1)  # edge_num * 1
        pred = sigmoid(dot_product_M)

        acc = accuracy(pred, train_label)
        loss = loss_fn(pred, train_label)
        if i % 1 == 0:
            print('epoch:{}, loss:{}, accuracy:{}'.format(i, loss, acc))

        loss.backward()
        optimizer.step()


def help_func_1():
    G = nx.karate_club_graph()
    pos_edge_list = graph_to_edge_list(G)
    pos_edge_index = edge_list_to_tensor(pos_edge_list)
    print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
    print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))

    # Sample 78 negative edges
    neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

    # Transform the negative edge list to tensor
    neg_edge_index = edge_list_to_tensor(neg_edge_list)
    print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

    # Which of following edges can be negative ones?
    test_edges = [(7, 1), (1, 33), (33, 22), (0, 4), (4, 2)]

    for test_e in test_edges:
        print(is_negative_edge(test_e, get_pos_edges(G)), end=', ')

    return pos_edge_index, neg_edge_index


def data_prepare():
    # edge data prepare
    pos_edge_index, neg_edge_index = help_func_1()
    train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    print('[Data Prepare] train_edge_shape', train_edge.shape)
    print(pos_edge_index.shape)

    # label data prepare
    pos_label = torch.ones(pos_edge_index.shape[1], )
    neg_label = torch.zeros(neg_edge_index.shape[1], )
    train_label = torch.cat([pos_label, neg_label], dim=0)
    print('[Data Prepare] train_label_shape: ', train_label.shape)

    return train_label, train_edge


def run_main():
    # settings
    torch.manual_seed(1)
    emb_dim = 16
    node_num = len(nx.karate_club_graph().nodes)

    # weight prepare
    emb = create_node_emb(node_num, emb_dim)
    # data prepare
    train_label, train_edge = data_prepare()

    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    visualize_emb(emb)
    train(emb, loss_fn, sigmoid, train_label, train_edge)
    visualize_emb(emb)


if __name__ == '__main__':
    run_main()

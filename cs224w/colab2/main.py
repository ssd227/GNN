import os
import torch
import copy
import pandas as pd
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from colab2.model import GCN


def get_num_classes(pyg_dataset):
    num_classes = pyg_dataset.num_classes
    return num_classes


def get_num_features(pyg_dataset):
    num_features = pyg_dataset.num_features
    return num_features


def get_graph_class(pyg_dataset, idx):
    label = pyg_dataset[idx].y
    return label


def get_graph_num_edges(pyg_dataset, idx):
    num_edges = dict()
    for (a, b) in pyg_dataset[idx].edge_index.t():
        ke = (a.item(), b.item()) if a <= b else (b.item(), a.item())
        num_edges[ke] = 1
    return len(num_edges)


def data_enzy():
    root = './enzymes'
    name = 'ENZYMES'

    pyg_dataset = TUDataset(root, name)
    num_classes = get_num_classes(pyg_dataset)
    num_features = get_num_features(pyg_dataset)
    print(pyg_dataset)
    print("{} dataset has {} classes".format(name, num_classes))
    print("{} dataset has {} features".format(name, num_features))

    graph_0 = pyg_dataset[0]
    print(graph_0)

    idx = 100
    label = get_graph_class(pyg_dataset, idx)
    print('Graph with index {} has label {}'.format(idx, label))

    idx = 200
    num_edges = get_graph_num_edges(pyg_dataset, idx)
    print('Graph with index {} has {} edges'.format(idx, num_edges))


def graph_num_features(data):
    num_features = data.num_features
    return num_features


def data_ogb():
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name,
                                     transform=T.ToSparseTensor())
    print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

    # Extract the graph
    data = dataset[0]
    print(data)

    num_features = graph_num_features(data)
    print('The graph has {} features'.format(num_features))


def data_prepare(config):
    device = config['device']
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name,
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    # Make the adjacency matrix to symmetric
    data.adj_t = data.adj_t.to_symmetric()

    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    return data, dataset, split_idx, train_idx


def train(model, data, train_idx, optimizer, loss_fn):
    model.train()  # 开启训练模式
    optimizer.zero_grad()

    y = model(data.x, data.adj_t)  # 直接上全部数据
    y_train = y[train_idx]
    y_target = data.y[train_idx].reshape(-1)

    # print(y_train.shape, y_train.dtype, y_target.shape, y_target.dtype, data.y.shape, data.y.dtype)

    loss = loss_fn(y_train, y_target)

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def mytest(model, data, split_idx, evaluator, save_model_results=False):
    # TODO: Implement a function that tests the model by
    # using the given split_idx and evaluator.

    model.eval()
    out = model(data.x, data.adj_t)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    if save_model_results:
        print("Saving Model Predictions")

        data = {}
        data['y_pred'] = y_pred.view(-1).cpu().detach().numpy()

        df = pd.DataFrame(data=data)
        # Save locally as csv
        df.to_csv('ogbn-arxiv_node.csv', sep=',', index=False)

    return train_acc, valid_acc, test_acc


def func_main(config):
    data, dataset, split_idx, train_idx = data_prepare(config)

    model = GCN(data.num_features, config['hidden_dim'],
                dataset.num_classes, config['num_layers'],
                config['dropout']).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')

    # reset the parameters to initial random value
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + config["epochs"]):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        # if epoch % 50 == 0:
        #     print('epoch', epoch, loss)
        result = mytest(model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')

    best_result = mytest(best_model, data, split_idx, evaluator, save_model_results=True)
    train_acc, valid_acc, test_acc = best_result
    print(f'Best model: '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    config = {
        'device': device,
        'num_layers': 4,
        'hidden_dim': 256,
        'dropout': 0,
        'lr': 0.01,
        'epochs': 100,
    }
    func_main(config)

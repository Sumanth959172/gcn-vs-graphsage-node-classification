import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import os


from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def run(dataset_name, hidden_dim, dropout):
    set_seed()

    dataset = Planetoid(root=f"data/Planetoid/{dataset_name}", name=dataset_name)
    data = dataset[0]

    model = GraphSAGE(
        dataset.num_features,
        hidden_dim,
        dataset.num_classes,
        dropout
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )

    start = time.time()
    for _ in range(150):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    train_time = time.time() - start

    val_acc, val_f1 = evaluate(model, data, data.val_mask)
    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    return {
        "model": "GraphSAGE(mean)",
        "dataset": dataset_name,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "train_time_sec": train_time,
    }


if __name__ == "__main__":
    datasets = ["Cora", "PubMed"]
    hidden_dims = [32, 64, 128]
    dropouts = [0.3, 0.5]

    results = []

    for d in datasets:
        for h in hidden_dims:
            for do in dropouts:
                print(f"GraphSAGE | {d} | hidden={h} | dropout={do}")
                res = run(d, h, do)
                results.append(res)
                print(res)

    out_dir = "experiments/exp06_graphsage_hparam"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)


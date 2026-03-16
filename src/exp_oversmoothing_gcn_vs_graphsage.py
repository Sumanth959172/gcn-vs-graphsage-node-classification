import os
import json
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import accuracy_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def drop_edges(edge_index, drop_ratio):
    if drop_ratio == 0:
        return edge_index.clone()

    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_ratio
    return edge_index[:, keep_mask]


def evaluate(model, data, split="test"):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    mask = {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }[split]

    return accuracy_score(
        data.y[mask].cpu().numpy(),
        pred[mask].cpu().numpy()
    )


def train_model(model, data, lr=0.01, weight_decay=5e-4, epochs=200, device="cpu"):
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -1.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, split="val")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run(dataset_name, edge_drop_rates=(0.0, 0.1, 0.2, 0.3), hidden_dim=64, epochs=200, seed=42):
    set_seed(seed)

    dataset = Planetoid(root=f"data/{dataset_name}", name=dataset_name)
    clean_data = dataset[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for model_name in ["GCN", "GraphSAGE"]:
        for drop_rate in edge_drop_rates:
            data = clean_data.clone()
            data.edge_index = drop_edges(clean_data.edge_index, drop_rate)

            if model_name == "GCN":
                model = GCN(dataset.num_features, hidden_dim, dataset.num_classes)
            else:
                model = GraphSAGE(dataset.num_features, hidden_dim, dataset.num_classes)

            model = train_model(model, data, epochs=epochs, device=device)
            test_acc = evaluate(model, data.to(device), split="test")

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "edge_drop_rate": drop_rate,
                "remaining_edges": int(data.edge_index.size(1)),
                "test_acc": test_acc,
                "seed": seed,
            }
            results.append(row)

            print(f"[{dataset_name}] {model_name} edge_drop={drop_rate} test_acc={test_acc:.4f}")

    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    all_results = []
    for ds in ["Cora", "PubMed"]:
        all_results.extend(run(ds))

    out_path = "results/edge_dropout_robustness.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {out_path}")

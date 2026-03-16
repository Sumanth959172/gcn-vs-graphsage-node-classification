import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 2.6+ compatibility
_torch_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = patched_load

from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import f1_score

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
from torch_geometric.utils import subgraph, to_undirected


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Models
# =========================================================
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x


# =========================================================
# Data preparation
# =========================================================
def build_subset_data(full_data, subset_size=10000, seed=42):
    set_seed(seed)

    num_nodes = full_data.num_nodes
    perm = torch.randperm(num_nodes)[:subset_size]
    perm = perm.sort().values

    # induced subgraph
    edge_index, _ = subgraph(
        subset=perm,
        edge_index=full_data.edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    edge_index = to_undirected(edge_index)

    x = full_data.x[perm].clone()
    y = full_data.y[perm].view(-1).clone()

    # feature normalization
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    subset_data = Data(x=x, edge_index=edge_index, y=y)

    # simple random split: 60 / 20 / 20
    n = subset_size
    order = torch.randperm(n)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    subset_data.train_mask = torch.zeros(n, dtype=torch.bool)
    subset_data.val_mask = torch.zeros(n, dtype=torch.bool)
    subset_data.test_mask = torch.zeros(n, dtype=torch.bool)

    subset_data.train_mask[train_idx] = True
    subset_data.val_mask[val_idx] = True
    subset_data.test_mask[test_idx] = True

    return subset_data


# =========================================================
# Training / evaluation
# =========================================================
@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return acc, macro_f1


def train_one_model(model, data, lr, weight_decay, max_epochs=300, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, val_f1 = evaluate(model, data, data.val_mask)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    return {
        "best_val_macro_f1": float(best_val_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "train_time_sec": float(train_time),
    }


def build_model(model_name, in_channels, out_channels):
    if model_name == "GCN":
        return GCNNet(in_channels, 128, out_channels, dropout=0.5)
    if model_name == "GraphSAGE":
        return GraphSAGENet(in_channels, 128, out_channels, dropout=0.5, aggr="mean")
    if model_name == "GAT":
        return GATNet(in_channels, 16, out_channels, heads=4, dropout=0.6)
    if model_name == "APPNP":
        return APPNPNet(in_channels, 128, out_channels, dropout=0.5, K=10, alpha=0.1)
    raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# Main
# =========================================================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading ogbn-arxiv...")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
    full_data = dataset[0]

    print("Building subset...")
    data = build_subset_data(full_data, subset_size=10000, seed=42).to(device)

    num_classes = int(data.y.max().item()) + 1
    in_channels = data.num_features

    results = {
        "dataset": "ogbn-arxiv subset",
        "subset_size": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.num_features),
        "num_classes": int(num_classes),
        "split": {
            "train": int(data.train_mask.sum().item()),
            "val": int(data.val_mask.sum().item()),
            "test": int(data.test_mask.sum().item()),
        },
        "models": {}
    }

    model_names = ["GCN", "GraphSAGE", "GAT", "APPNP"]

    for model_name in model_names:
        print(f"\nRunning {model_name}...")

        model = build_model(model_name, in_channels, num_classes).to(device)

        metrics = train_one_model(
            model=model,
            data=data,
            lr=0.01,
            weight_decay=5e-4,
            max_epochs=300,
            patience=50,
        )

        results["models"][model_name] = metrics

        print(
            f"{model_name} | "
            f"test_acc={metrics['test_acc']:.4f} | "
            f"test_macro_f1={metrics['test_macro_f1']:.4f} | "
            f"time={metrics['train_time_sec']:.1f}s"
        )

    out_dir = "experiments/ogbn_arxiv_subset"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
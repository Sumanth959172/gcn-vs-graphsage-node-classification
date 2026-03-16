import os
import json
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Models
# -----------------------------
class GCNDepthEmb(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.6):
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.out_conv = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_last_hidden=False):

        last_hidden = None

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

            last_hidden = x

            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.out_conv(x, edge_index)

        if return_last_hidden:
            return out, last_hidden

        return out


class GraphSAGEDepthEmb(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.6):
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(depth - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.out_conv = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_last_hidden=False):

        last_hidden = None

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

            last_hidden = x

            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.out_conv(x, edge_index)

        if return_last_hidden:
            return out, last_hidden

        return out


# -----------------------------
# Train
# -----------------------------
@torch.no_grad()
def evaluate(model, data, mask):

    model.eval()

    logits = model(data.x, data.edge_index)

    preds = logits[mask].argmax(dim=1)
    labels = data.y[mask]

    acc = (preds == labels).float().mean().item()

    return acc


def train(model, data, epochs=400):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val = 0
    best_state = None

    patience = 100
    counter = 0

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()

        optimizer.step()

        val_acc = evaluate(model, data, data.val_mask)

        if val_acc > best_val:

            best_val = val_acc
            best_state = model.state_dict()

            counter = 0

        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_state)

    return best_val


# -----------------------------
# Plotting
# -----------------------------
def plot_embedding(emb, labels, out_file, title, method="pca"):

    if method == "pca":
        reducer = PCA(n_components=2)

    else:
        reducer = TSNE(n_components=2, random_state=42)

    z = reducer.fit_transform(emb)

    plt.figure(figsize=(6,5))

    plt.scatter(z[:,0], z[:,1], c=labels, s=8)

    plt.title(title)

    plt.tight_layout()

    plt.savefig(out_file, dpi=220)

    plt.close()


# -----------------------------
# Dataset
# -----------------------------
def load_dataset(name, device):

    dataset = Planetoid(root=f"data/Planetoid/{name}", name=name)

    data = dataset[0]

    x = data.x

    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    return dataset, data.to(device)


# -----------------------------
# Experiment
# -----------------------------
def run_single(dataset_name, model_name, depth, device):

    dataset, data = load_dataset(dataset_name, device)

    hidden = 256

    if model_name == "GCN":

        model = GCNDepthEmb(
            dataset.num_features,
            hidden,
            dataset.num_classes,
            depth
        ).to(device)

    else:

        model = GraphSAGEDepthEmb(
            dataset.num_features,
            hidden,
            dataset.num_classes,
            depth
        ).to(device)

    best_val = train(model, data)

    model.eval()

    _, emb = model(data.x, data.edge_index, return_last_hidden=True)

    emb = emb.cpu().detach().numpy()

    labels = data.y.cpu().numpy()

    return emb, labels, best_val


# -----------------------------
# Main
# -----------------------------
def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = ["Cora", "PubMed"]

    models = ["GCN", "GraphSAGE"]

    depths = [2, 4]

    out_root = os.path.join(
        "experiments",
        "embedding_plots",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    os.makedirs(out_root, exist_ok=True)

    results = []

    for dataset in datasets:

        for model in models:

            for depth in depths:

                print(f"Running {dataset} | {model} | depth={depth}")

                emb, labels, val_acc = run_single(
                    dataset,
                    model,
                    depth,
                    device
                )

                run_name = f"{dataset}_{model}_depth{depth}"

                run_dir = os.path.join(out_root, run_name)

                os.makedirs(run_dir, exist_ok=True)

                plot_embedding(
                    emb,
                    labels,
                    os.path.join(run_dir, "pca.png"),
                    f"{dataset} {model} depth={depth} PCA",
                    "pca"
                )

                plot_embedding(
                    emb,
                    labels,
                    os.path.join(run_dir, "tsne.png"),
                    f"{dataset} {model} depth={depth} tSNE",
                    "tsne"
                )

                results.append({
                    "dataset": dataset,
                    "model": model,
                    "depth": depth,
                    "val_acc": float(val_acc)
                })

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll plots saved to:")
    print(out_root)


if __name__ == "__main__":
    main()
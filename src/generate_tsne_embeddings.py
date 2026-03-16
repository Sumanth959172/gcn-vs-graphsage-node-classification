import os

import json

import random

from pathlib import Path
 
import numpy as np

import matplotlib.pyplot as plt
 
import torch

import torch.nn as nn

import torch.nn.functional as F
 
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid

from torch_geometric.nn import GCNConv, SAGEConv

from torch_geometric.utils import subgraph
 
# OGB

from ogb.nodeproppred import PygNodePropPredDataset
 
 
# =========================================================

# PyTorch 2.6+ compatibility for OGB processed PyG files

# =========================================================

_torch_load = torch.load
 
 
def patched_load(*args, **kwargs):

    kwargs.setdefault("weights_only", False)

    return _torch_load(*args, **kwargs)
 
 
torch.load = patched_load
 
 
# =========================================================

# CONFIG

# =========================================================

OUT_DIR = "results/tsne_embeddings"
 
RUNS = [

    {"dataset": "Cora", "model": "GCN", "depth": 2},

    {"dataset": "Cora", "model": "GCN", "depth": 4},

    {"dataset": "Cora", "model": "GraphSAGE", "depth": 2},

    {"dataset": "Cora", "model": "GraphSAGE", "depth": 4},

    {"dataset": "PubMed", "model": "GCN", "depth": 2},

    {"dataset": "PubMed", "model": "GCN", "depth": 4},

    {"dataset": "PubMed", "model": "GraphSAGE", "depth": 2},

    {"dataset": "PubMed", "model": "GraphSAGE", "depth": 4},
 
    # Added OGBN-Arxiv

    {"dataset": "ogbn-arxiv", "model": "GCN", "depth": 2},

    {"dataset": "ogbn-arxiv", "model": "GCN", "depth": 4},

    {"dataset": "ogbn-arxiv", "model": "GraphSAGE", "depth": 2},

    {"dataset": "ogbn-arxiv", "model": "GraphSAGE", "depth": 4},

]
 
CFG = {

    "hidden_dim": 256,

    "dropout": 0.6,

    "lr": 0.01,

    "weight_decay": 5e-4,

    "epochs": 500,

    "patience": 100,

    "min_delta": 1e-4,

    "use_bn": True,

    "sage_aggr": "mean",

    "seed": 42,
 
    # OGB subset settings

    # Set to None to use the full ogbn-arxiv graph

    "ogbn_subset_num_nodes": 10000,

}
 
 
# =========================================================

# HELPERS

# =========================================================

def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
 
 
def ensure_dir(path: str):

    os.makedirs(path, exist_ok=True)
 
 
def standardize_features(x: torch.Tensor) -> torch.Tensor:

    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
 
 
def make_mask(num_nodes: int, idx: torch.Tensor) -> torch.Tensor:

    mask = torch.zeros(num_nodes, dtype=torch.bool)

    mask[idx] = True

    return mask
 
 
def sample_ogbn_subset(data, split_idx, num_nodes: int, seed: int = 42):

    """

    Create an induced subgraph from OGBN-Arxiv using a random subset of nodes.

    Split masks are re-created by intersecting original split indices with the subset.

    """

    if num_nodes is None or num_nodes >= data.num_nodes:

        return data
 
    g = torch.Generator()

    g.manual_seed(seed)

    perm = torch.randperm(data.num_nodes, generator=g)

    subset_nodes = perm[:num_nodes]

    subset_nodes, _ = torch.sort(subset_nodes)
 
    # Induced subgraph

    edge_index, _ = subgraph(subset_nodes, data.edge_index, relabel_nodes=True)
 
    new_data = data.clone()

    new_data.x = data.x[subset_nodes]

    new_data.y = data.y[subset_nodes]

    new_data.edge_index = edge_index

    new_data.num_nodes = subset_nodes.numel()
 
    # Map old node ids -> new node ids

    old_to_new = -torch.ones(data.num_nodes, dtype=torch.long)

    old_to_new[subset_nodes] = torch.arange(subset_nodes.numel())
 
    def remap_split(split_tensor: torch.Tensor) -> torch.Tensor:

        split_tensor = split_tensor.view(-1)

        keep = old_to_new[split_tensor] >= 0

        return old_to_new[split_tensor[keep]]
 
    train_idx = remap_split(split_idx["train"])

    valid_idx = remap_split(split_idx["valid"])

    test_idx = remap_split(split_idx["test"])
 
    new_data.train_mask = make_mask(new_data.num_nodes, train_idx)

    new_data.val_mask = make_mask(new_data.num_nodes, valid_idx)

    new_data.test_mask = make_mask(new_data.num_nodes, test_idx)
 
    return new_data
 
 
def load_planetoid_dataset(name: str, device: torch.device):

    dataset = Planetoid(root=f"data/Planetoid/{name}", name=name)

    data = dataset[0]
 
    data.x = standardize_features(data.x)
 
    # Ensure labels are 1D

    if data.y.dim() > 1:

        data.y = data.y.view(-1)
 
    return dataset, data.to(device)
 
 
def load_ogbn_arxiv(device: torch.device, subset_num_nodes=None, seed: int = 42):

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data/OGB")

    data = dataset[0]
 
    # OGB labels often come as shape [N, 1]

    data.y = data.y.view(-1)

    data.x = standardize_features(data.x)
 
    split_idx = dataset.get_idx_split()
 
    if subset_num_nodes is not None:

        data = sample_ogbn_subset(data, split_idx, subset_num_nodes, seed=seed)

    else:

        data.train_mask = make_mask(data.num_nodes, split_idx["train"].view(-1))

        data.val_mask = make_mask(data.num_nodes, split_idx["valid"].view(-1))

        data.test_mask = make_mask(data.num_nodes, split_idx["test"].view(-1))
 
    return dataset, data.to(device)
 
 
def load_dataset(name: str, device: torch.device):

    name_lower = name.lower()
 
    if name_lower in {"cora", "pubmed", "citeseer"}:

        return load_planetoid_dataset(name, device)
 
    if name_lower == "ogbn-arxiv":

        return load_ogbn_arxiv(

            device=device,

            subset_num_nodes=CFG["ogbn_subset_num_nodes"],

            seed=CFG["seed"],

        )
 
    raise ValueError(f"Unsupported dataset: {name}")
 
 
@torch.no_grad()

def evaluate_acc(model, data, mask):

    model.eval()

    logits = model(data.x, data.edge_index)

    preds = logits[mask].argmax(dim=1)

    labels = data.y[mask]

    return float((preds == labels).float().mean().item())
 
 
# =========================================================

# MODELS

# =========================================================

class GCNDepthEmb(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.6, use_bn=True):

        super().__init__()

        assert depth >= 2

        self.dropout = float(dropout)

        self.use_bn = bool(use_bn)
 
        self.convs = nn.ModuleList()

        self.bns = nn.ModuleList()
 
        self.convs.append(GCNConv(in_channels, hidden_channels))

        if self.use_bn:

            self.bns.append(nn.BatchNorm1d(hidden_channels))
 
        for _ in range(depth - 2):

            self.convs.append(GCNConv(hidden_channels, hidden_channels))

            if self.use_bn:

                self.bns.append(nn.BatchNorm1d(hidden_channels))
 
        self.out_conv = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index, return_last_hidden=False):

        last_hidden = None
 
        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index)

            if self.use_bn:

                x = self.bns[i](x)

            x = F.relu(x)

            last_hidden = x

            x = F.dropout(x, p=self.dropout, training=self.training)
 
        out = self.out_conv(x, edge_index)
 
        if return_last_hidden:

            return out, last_hidden

        return out
 
 
class GraphSAGEDepthEmb(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.6, use_bn=True, aggr="mean"):

        super().__init__()

        assert depth >= 2

        self.dropout = float(dropout)

        self.use_bn = bool(use_bn)
 
        self.convs = nn.ModuleList()

        self.bns = nn.ModuleList()
 
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

        if self.use_bn:

            self.bns.append(nn.BatchNorm1d(hidden_channels))
 
        for _ in range(depth - 2):

            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

            if self.use_bn:

                self.bns.append(nn.BatchNorm1d(hidden_channels))
 
        self.out_conv = SAGEConv(hidden_channels, out_channels, aggr=aggr)
 
    def forward(self, x, edge_index, return_last_hidden=False):

        last_hidden = None
 
        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index)

            if self.use_bn:

                x = self.bns[i](x)

            x = F.relu(x)

            last_hidden = x

            x = F.dropout(x, p=self.dropout, training=self.training)
 
        out = self.out_conv(x, edge_index)
 
        if return_last_hidden:

            return out, last_hidden

        return out
 
 
# =========================================================

# TRAIN

# =========================================================

def train(model, data, epochs=500, lr=0.01, weight_decay=5e-4, patience=100, min_delta=1e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    best_val_acc = -1.0

    best_state = None

    counter = 0
 
    for _ in range(epochs):

        model.train()

        optimizer.zero_grad()
 
        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
 
        val_acc = evaluate_acc(model, data, data.val_mask)
 
        if val_acc > best_val_acc + min_delta:

            best_val_acc = val_acc

            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            counter = 0

        else:

            counter += 1
 
        if counter >= patience:

            break
 
    if best_state is not None:

        model.load_state_dict(best_state)
 
    return float(best_val_acc)
 
 
# =========================================================

# PLOT

# =========================================================

def save_tsne_plot(emb, labels, out_file, title):

    tsne = TSNE(

        n_components=2,

        random_state=42,

        init="pca",

        learning_rate="auto",

        perplexity=30,

    )
 
    z = tsne.fit_transform(emb)
 
    plt.figure(figsize=(7, 5))

    plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)

    plt.title(title)

    plt.tight_layout()

    plt.savefig(out_file, dpi=220)

    plt.close()
 
 
def run_single(run_cfg, device):

    dataset_name = run_cfg["dataset"]

    model_name = run_cfg["model"]

    depth = run_cfg["depth"]
 
    dataset, data = load_dataset(dataset_name, device)
 
    if model_name == "GCN":

        model = GCNDepthEmb(

            in_channels=dataset.num_features,

            hidden_channels=CFG["hidden_dim"],

            out_channels=dataset.num_classes,

            depth=depth,

            dropout=CFG["dropout"],

            use_bn=CFG["use_bn"],

        ).to(device)

    elif model_name == "GraphSAGE":

        model = GraphSAGEDepthEmb(

            in_channels=dataset.num_features,

            hidden_channels=CFG["hidden_dim"],

            out_channels=dataset.num_classes,

            depth=depth,

            dropout=CFG["dropout"],

            use_bn=CFG["use_bn"],

            aggr=CFG["sage_aggr"],

        ).to(device)

    else:

        raise ValueError(f"Unsupported model: {model_name}")
 
    best_val_acc = train(

        model=model,

        data=data,

        epochs=CFG["epochs"],

        lr=CFG["lr"],

        weight_decay=CFG["weight_decay"],

        patience=CFG["patience"],

        min_delta=CFG["min_delta"],

    )
 
    model.eval()

    _, hidden = model(data.x, data.edge_index, return_last_hidden=True)
 
    emb = hidden.detach().cpu().numpy()

    labels = data.y.detach().cpu().numpy()
 
    run_name = f"{dataset_name}_{model_name}_depth{depth}"

    run_dir = os.path.join(OUT_DIR, run_name)

    ensure_dir(run_dir)
 
    save_tsne_plot(

        emb=emb,

        labels=labels,

        out_file=os.path.join(run_dir, "tsne.png"),

        title=f"{dataset_name} | {model_name} | depth={depth} | t-SNE",

    )
 
    meta = {

        "dataset": dataset_name,

        "model": model_name,

        "depth": depth,

        "best_val_acc": best_val_acc,

        "saved_dir": run_dir,

        "num_nodes": int(data.num_nodes),

        "num_edges": int(data.edge_index.size(1)),

        "ogbn_subset_num_nodes": CFG["ogbn_subset_num_nodes"] if dataset_name.lower() == "ogbn-arxiv" else None,

    }
 
    with open(os.path.join(run_dir, "meta.json"), "w") as f:

        json.dump(meta, f, indent=2)
 
    print(f"Saved: {run_dir}")
 
 
def main():

    ensure_dir(OUT_DIR)

    set_seed(CFG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    for run_cfg in RUNS:

        run_single(run_cfg, device)
 
    print(f"All t-SNE embeddings saved to: {OUT_DIR}")
 
 
if __name__ == "__main__":

    main()
 
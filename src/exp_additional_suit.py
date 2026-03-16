import os
import json
import time
import math
import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 2.6+ compatibility
_torch_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = patched_load

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
from torch_geometric.utils import subgraph, to_undirected

from ogb.nodeproppred import PygNodePropPredDataset


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "seeds": [0, 1, 2],
    "out_dir": "experiments/additional_suite",
    "run_experiments": {
        "reduced_training_data": True,
        "inductive_style_split": True,
        "memory_usage": True,
        "classwise_f1": True,
        "confusion_matrix": True,
        "confidence_analysis": True,
        "hidden_dim_ablation": True,
        "dropout_ablation": True,
        "weight_decay_ablation": True,
        "convergence_analysis": True,
    },
    "datasets": {
        "Cora": True,
        "PubMed": True,
        "ogbn-arxiv-subset": True,
    },
    "ogbn_subset_size": 10000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

DEVICE = torch.device(CONFIG["device"])


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# MODELS
# =========================================================
class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, use_bn=True):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.bn1 = nn.BatchNorm1d(hidden_channels) if use_bn else None

    def forward(self, x, edge_index, return_hidden=False):
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        h = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.conv2(x, edge_index)
        if return_hidden:
            return out, h
        return out


class GraphSAGE2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, use_bn=True, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.bn1 = nn.BatchNorm1d(hidden_channels) if use_bn else None

    def forward(self, x, edge_index, return_hidden=False):
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        h = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.conv2(x, edge_index)
        if return_hidden:
            return out, h
        return out


class GAT2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, return_hidden=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        h = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.gat2(x, edge_index)
        if return_hidden:
            return out, h
        return out


class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, return_hidden=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        h = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        out = self.prop(x, edge_index)
        if return_hidden:
            return out, h
        return out


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clone_data(data: Data):
    out = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
    )
    for attr in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, attr):
            setattr(out, attr, getattr(data, attr).clone())
    return out


def normalize_features(x):
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) == 0:
        return None, None
    xs = np.array(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def maybe_get_memory_mb():
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return None


def clear_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =========================================================
# DATA LOADING
# =========================================================
def load_planetoid_dataset(name: str):
    dataset = Planetoid(root=f"data/Planetoid/{name}", name=name)
    data = dataset[0]
    data.x = normalize_features(data.x)
    return data, dataset.num_features, dataset.num_classes


def build_ogbn_subset(subset_size=10000, seed=42):
    set_seed(seed)
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
    full_data = dataset[0]

    perm = torch.randperm(full_data.num_nodes)[:subset_size]
    perm = perm.sort().values

    edge_index, _ = subgraph(
        subset=perm,
        edge_index=full_data.edge_index,
        relabel_nodes=True,
        num_nodes=full_data.num_nodes,
    )
    edge_index = to_undirected(edge_index)

    x = full_data.x[perm].clone()
    y = full_data.y[perm].view(-1).clone()
    x = normalize_features(x)

    data = Data(x=x, edge_index=edge_index, y=y)

    order = torch.randperm(subset_size)
    n_train = int(0.6 * subset_size)
    n_val = int(0.2 * subset_size)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    data.train_mask = torch.zeros(subset_size, dtype=torch.bool)
    data.val_mask = torch.zeros(subset_size, dtype=torch.bool)
    data.test_mask = torch.zeros(subset_size, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    num_classes = int(y.max().item()) + 1
    return data, data.num_features, num_classes


def load_all_datasets():
    loaded = {}

    if CONFIG["datasets"]["Cora"]:
        loaded["Cora"] = load_planetoid_dataset("Cora")

    if CONFIG["datasets"]["PubMed"]:
        loaded["PubMed"] = load_planetoid_dataset("PubMed")

    if CONFIG["datasets"]["ogbn-arxiv-subset"]:
        loaded["ogbn-arxiv-subset"] = build_ogbn_subset(
            subset_size=CONFIG["ogbn_subset_size"],
            seed=42
        )

    return loaded


# =========================================================
# MODEL BUILDER
# =========================================================
def build_model(model_name, in_channels, out_channels, hidden_dim=128, dropout=0.5, weight_decay=None):
    if model_name == "GCN":
        return GCN2(in_channels, hidden_dim, out_channels, dropout=dropout, use_bn=True)
    if model_name == "GraphSAGE":
        return GraphSAGE2(in_channels, hidden_dim, out_channels, dropout=dropout, use_bn=True, aggr="mean")
    if model_name == "GAT":
        gat_hidden = 16 if hidden_dim < 64 else 16
        return GAT2(in_channels, gat_hidden, out_channels, heads=4, dropout=max(0.5, dropout))
    if model_name == "APPNP":
        return APPNPNet(in_channels, hidden_dim, out_channels, dropout=dropout, K=10, alpha=0.1)
    raise ValueError(model_name)


# =========================================================
# TRAIN / EVAL
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


@torch.no_grad()
def predict_full(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    confs = probs.max(dim=1).values
    return logits, probs, preds, confs


def train_one_run(
    model,
    data,
    lr=0.01,
    weight_decay=5e-4,
    max_epochs=300,
    patience=50,
    min_delta=1e-4,
    track_history=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    clear_memory_stats()
    t0 = time.time()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    } if track_history else None

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        val_acc, val_f1 = evaluate(model, data, data.val_mask)

        if track_history:
            history["epoch"].append(int(epoch))
            history["train_loss"].append(float(loss.item()))
            history["val_acc"].append(float(val_acc))
            history["val_macro_f1"].append(float(val_f1))

        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    train_time = time.time() - t0
    peak_mem_mb = maybe_get_memory_mb()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    result = {
        "best_val_macro_f1": float(best_val_f1),
        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(test_f1),
        "train_time_sec": float(train_time),
        "peak_memory_mb": peak_mem_mb,
    }

    if track_history:
        result["history"] = history

    return result


# =========================================================
# SPLIT / PERTURBATION UTILS
# =========================================================
def reduce_training_mask(data, fraction, seed=42):
    new_data = clone_data(data)
    set_seed(seed)
    train_idx = new_data.train_mask.nonzero(as_tuple=False).view(-1)
    k = max(1, int(len(train_idx) * fraction))
    perm = torch.randperm(len(train_idx))[:k]
    chosen = train_idx[perm]

    new_data.train_mask = torch.zeros_like(new_data.train_mask)
    new_data.train_mask[chosen] = True
    return new_data


def make_inductive_style_split(data, seed=42):
    """
    Simple inductive-style approximation:
    - randomly split nodes into train/val/test
    - train graph only keeps train+val nodes
    - test nodes remain unseen during training
    - evaluation on full node features but no train-time access to test nodes in graph
    """
    set_seed(seed)
    n = data.num_nodes
    order = torch.randperm(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    new_data = clone_data(data)
    new_data.train_mask = torch.zeros(n, dtype=torch.bool)
    new_data.val_mask = torch.zeros(n, dtype=torch.bool)
    new_data.test_mask = torch.zeros(n, dtype=torch.bool)
    new_data.train_mask[train_idx] = True
    new_data.val_mask[val_idx] = True
    new_data.test_mask[test_idx] = True

    visible = torch.cat([train_idx, val_idx]).unique()
    edge_index, _ = subgraph(
        subset=visible,
        edge_index=new_data.edge_index,
        relabel_nodes=False,
        num_nodes=n
    )
    new_data.edge_index = edge_index
    return new_data


# =========================================================
# EXPERIMENT 1: REDUCED TRAINING DATA
# =========================================================
def exp_reduced_training_data(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    fractions = [1.0, 0.5, 0.25, 0.1]
    models = ["GCN", "GraphSAGE", "GAT", "APPNP"]

    for dataset_name, (base_data, in_ch, out_ch) in datasets.items():
        for model_name in models:
            for frac in fractions:
                per_seed = []
                for seed in CONFIG["seeds"]:
                    set_seed(seed)
                    data = reduce_training_mask(base_data, frac, seed=seed).to(DEVICE)
                    model = build_model(model_name, in_ch, out_ch).to(DEVICE)
                    m = train_one_run(model, data, lr=0.01, weight_decay=5e-4)
                    m["seed"] = int(seed)
                    per_seed.append(m)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "train_fraction": frac,
                    "per_seed": per_seed,
                    "summary": {
                        "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                        "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                        "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                        "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                    }
                })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 2: INDUCTIVE-STYLE SPLIT
# =========================================================
def exp_inductive_style_split(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    models = ["GCN", "GraphSAGE"]

    for dataset_name, (base_data, in_ch, out_ch) in datasets.items():
        for model_name in models:
            per_seed = []
            for seed in CONFIG["seeds"]:
                set_seed(seed)
                data = make_inductive_style_split(base_data, seed=seed).to(DEVICE)
                model = build_model(model_name, in_ch, out_ch).to(DEVICE)
                m = train_one_run(model, data, lr=0.01, weight_decay=5e-4)
                m["seed"] = int(seed)
                per_seed.append(m)

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "per_seed": per_seed,
                "summary": {
                    "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                    "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                    "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                    "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                }
            })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 3: MEMORY USAGE
# =========================================================
def exp_memory_usage(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    models = ["GCN", "GraphSAGE", "GAT", "APPNP"]

    for dataset_name, (base_data, in_ch, out_ch) in datasets.items():
        for model_name in models:
            per_seed = []
            for seed in CONFIG["seeds"]:
                set_seed(seed)
                data = clone_data(base_data).to(DEVICE)
                model = build_model(model_name, in_ch, out_ch).to(DEVICE)
                m = train_one_run(model, data, lr=0.01, weight_decay=5e-4)
                m["seed"] = int(seed)
                per_seed.append(m)

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "num_parameters": int(count_parameters(build_model(model_name, in_ch, out_ch))),
                "per_seed": per_seed,
                "summary": {
                    "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                    "train_time_sec_mean": mean_std([x["train_time_sec"] for x in per_seed])[0],
                    "peak_memory_mb_mean": mean_std([x["peak_memory_mb"] for x in per_seed])[0],
                    "peak_memory_mb_std": mean_std([x["peak_memory_mb"] for x in per_seed])[1],
                }
            })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 4: CLASS-WISE F1
# =========================================================
def exp_classwise_f1(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed"]}
    models = ["GCN", "GraphSAGE"]

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            set_seed(42)
            data = clone_data(base_data).to(DEVICE)
            model = build_model(model_name, in_ch, out_ch).to(DEVICE)
            train_one_run(model, data, lr=0.01, weight_decay=5e-4)

            _, _, preds, _ = predict_full(model, data)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = preds[data.test_mask].cpu().numpy()

            p, r, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            rows = []
            for c in range(len(f1)):
                rows.append({
                    "class": int(c),
                    "precision": float(p[c]),
                    "recall": float(r[c]),
                    "f1": float(f1[c]),
                    "support": int(support[c]),
                })

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "classwise_metrics": rows,
            })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 5: CONFUSION MATRIX
# =========================================================
def exp_confusion_matrix(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed"]}
    models = ["GCN", "GraphSAGE"]

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            set_seed(42)
            data = clone_data(base_data).to(DEVICE)
            model = build_model(model_name, in_ch, out_ch).to(DEVICE)
            train_one_run(model, data, lr=0.01, weight_decay=5e-4)

            _, _, preds, _ = predict_full(model, data)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = preds[data.test_mask].cpu().numpy()

            cm = confusion_matrix(y_true, y_pred)

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "confusion_matrix": cm.tolist(),
            })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 6: CONFIDENCE ANALYSIS
# =========================================================
def exp_confidence_analysis(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed"]}
    models = ["GCN", "GraphSAGE"]

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            set_seed(42)
            data = clone_data(base_data).to(DEVICE)
            model = build_model(model_name, in_ch, out_ch).to(DEVICE)
            train_one_run(model, data, lr=0.01, weight_decay=5e-4)

            _, _, preds, confs = predict_full(model, data)

            y_true = data.y[data.test_mask]
            y_pred = preds[data.test_mask]
            test_conf = confs[data.test_mask]

            correct_mask = (y_true == y_pred)

            correct_conf = test_conf[correct_mask].cpu().numpy()
            wrong_conf = test_conf[~correct_mask].cpu().numpy()

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "avg_conf_correct": float(correct_conf.mean()) if len(correct_conf) > 0 else None,
                "avg_conf_wrong": float(wrong_conf.mean()) if len(wrong_conf) > 0 else None,
                "num_correct": int(correct_mask.sum().item()),
                "num_wrong": int((~correct_mask).sum().item()),
            })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 7: HIDDEN DIM ABLATION
# =========================================================
def exp_hidden_dim_ablation(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    dims = [32, 64, 128, 256]
    models = ["GCN", "GraphSAGE"]

    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed", "ogbn-arxiv-subset"]}

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            for hidden_dim in dims:
                per_seed = []
                for seed in CONFIG["seeds"]:
                    set_seed(seed)
                    data = clone_data(base_data).to(DEVICE)
                    model = build_model(model_name, in_ch, out_ch, hidden_dim=hidden_dim).to(DEVICE)
                    m = train_one_run(model, data, lr=0.01, weight_decay=5e-4)
                    m["seed"] = int(seed)
                    per_seed.append(m)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "hidden_dim": int(hidden_dim),
                    "per_seed": per_seed,
                    "summary": {
                        "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                        "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                        "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                        "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                    }
                })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 8: DROPOUT ABLATION
# =========================================================
def exp_dropout_ablation(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    dropouts = [0.0, 0.3, 0.5, 0.7]
    models = ["GCN", "GraphSAGE"]

    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed", "ogbn-arxiv-subset"]}

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            for dropout in dropouts:
                per_seed = []
                for seed in CONFIG["seeds"]:
                    set_seed(seed)
                    data = clone_data(base_data).to(DEVICE)
                    model = build_model(model_name, in_ch, out_ch, hidden_dim=128, dropout=dropout).to(DEVICE)
                    m = train_one_run(model, data, lr=0.01, weight_decay=5e-4)
                    m["seed"] = int(seed)
                    per_seed.append(m)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "dropout": float(dropout),
                    "per_seed": per_seed,
                    "summary": {
                        "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                        "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                        "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                        "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                    }
                })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 9: WEIGHT DECAY ABLATION
# =========================================================
def exp_weight_decay_ablation(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    wds = [0.0, 1e-4, 5e-4, 1e-3]
    models = ["GCN", "GraphSAGE"]

    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed", "ogbn-arxiv-subset"]}

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            for wd in wds:
                per_seed = []
                for seed in CONFIG["seeds"]:
                    set_seed(seed)
                    data = clone_data(base_data).to(DEVICE)
                    model = build_model(model_name, in_ch, out_ch).to(DEVICE)
                    m = train_one_run(model, data, lr=0.01, weight_decay=wd)
                    m["seed"] = int(seed)
                    per_seed.append(m)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "weight_decay": float(wd),
                    "per_seed": per_seed,
                    "summary": {
                        "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                        "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                        "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                        "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                    }
                })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# EXPERIMENT 10: CONVERGENCE ANALYSIS
# =========================================================
def exp_convergence_analysis(datasets, out_dir):
    ensure_dir(out_dir)
    results = []

    models = ["GCN", "GraphSAGE"]
    selected = {k: v for k, v in datasets.items() if k in ["Cora", "PubMed", "ogbn-arxiv-subset"]}

    for dataset_name, (base_data, in_ch, out_ch) in selected.items():
        for model_name in models:
            per_seed = []
            for seed in CONFIG["seeds"]:
                set_seed(seed)
                data = clone_data(base_data).to(DEVICE)
                model = build_model(model_name, in_ch, out_ch).to(DEVICE)
                m = train_one_run(
                    model,
                    data,
                    lr=0.01,
                    weight_decay=5e-4,
                    max_epochs=300,
                    patience=50,
                    track_history=True,
                )
                m["seed"] = int(seed)
                per_seed.append(m)

            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "per_seed": per_seed,
            })

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(CONFIG["out_dir"])
    datasets = load_all_datasets()

    if CONFIG["run_experiments"]["reduced_training_data"]:
        print("Running reduced training data experiment...")
        exp_reduced_training_data(datasets, os.path.join(CONFIG["out_dir"], "reduced_training_data"))

    if CONFIG["run_experiments"]["inductive_style_split"]:
        print("Running inductive-style split experiment...")
        exp_inductive_style_split(datasets, os.path.join(CONFIG["out_dir"], "inductive_style_split"))

    if CONFIG["run_experiments"]["memory_usage"]:
        print("Running memory usage experiment...")
        exp_memory_usage(datasets, os.path.join(CONFIG["out_dir"], "memory_usage"))

    if CONFIG["run_experiments"]["classwise_f1"]:
        print("Running class-wise F1 experiment...")
        exp_classwise_f1(datasets, os.path.join(CONFIG["out_dir"], "classwise_f1"))

    if CONFIG["run_experiments"]["confusion_matrix"]:
        print("Running confusion matrix experiment...")
        exp_confusion_matrix(datasets, os.path.join(CONFIG["out_dir"], "confusion_matrix"))

    if CONFIG["run_experiments"]["confidence_analysis"]:
        print("Running confidence analysis experiment...")
        exp_confidence_analysis(datasets, os.path.join(CONFIG["out_dir"], "confidence_analysis"))

    if CONFIG["run_experiments"]["hidden_dim_ablation"]:
        print("Running hidden dimension ablation...")
        exp_hidden_dim_ablation(datasets, os.path.join(CONFIG["out_dir"], "hidden_dim_ablation"))

    if CONFIG["run_experiments"]["dropout_ablation"]:
        print("Running dropout ablation...")
        exp_dropout_ablation(datasets, os.path.join(CONFIG["out_dir"], "dropout_ablation"))

    if CONFIG["run_experiments"]["weight_decay_ablation"]:
        print("Running weight decay ablation...")
        exp_weight_decay_ablation(datasets, os.path.join(CONFIG["out_dir"], "weight_decay_ablation"))

    if CONFIG["run_experiments"]["convergence_analysis"]:
        print("Running convergence analysis...")
        exp_convergence_analysis(datasets, os.path.join(CONFIG["out_dir"], "convergence_analysis"))

    print(f"\nAll selected experiments finished. Results saved under: {CONFIG['out_dir']}")


if __name__ == "__main__":
    main()
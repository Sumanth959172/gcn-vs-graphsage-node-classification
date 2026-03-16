
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

from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
from torch_geometric.utils import subgraph, to_undirected, degree, dropout_edge


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
class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, use_bn=True):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.bn1 = nn.BatchNorm1d(hidden_channels) if use_bn else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, use_bn=True, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.bn1 = nn.BatchNorm1d(hidden_channels) if use_bn else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT2(nn.Module):
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


class GCNDepthEmb(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.5, use_bn=True):
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

    def forward(self, x, edge_index, return_hidden=False):
        hidden = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            hidden.append(x.detach())
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.out_conv(x, edge_index)
        if return_hidden:
            return out, hidden
        return out


class SAGEDepthEmb(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.5, use_bn=True, aggr="mean"):
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

    def forward(self, x, edge_index, return_hidden=False):
        hidden = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            hidden.append(x.detach())
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.out_conv(x, edge_index)
        if return_hidden:
            return out, hidden
        return out


# =========================================================
# Helpers
# =========================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) == 0:
        return None, None
    xs = np.array(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


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
def evaluate_subset(model, data, node_idx):
    model.eval()
    if len(node_idx) == 0:
        return {"num_nodes": 0, "acc": None, "macro_f1": None}

    node_idx = torch.tensor(node_idx, dtype=torch.long, device=data.x.device)
    logits = model(data.x, data.edge_index)
    preds = logits[node_idx].argmax(dim=1).cpu().numpy()
    labels = data.y[node_idx].cpu().numpy()
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return {"num_nodes": int(len(node_idx)), "acc": acc, "macro_f1": macro_f1}


def average_cosine_similarity(x: torch.Tensor, sample_size: int = 2000) -> float:
    x = F.normalize(x, p=2, dim=1)
    n = x.size(0)
    if n < 2:
        return 0.0
    idx1 = torch.randint(0, n, (sample_size,), device=x.device)
    idx2 = torch.randint(0, n, (sample_size,), device=x.device)
    sims = (x[idx1] * x[idx2]).sum(dim=1)
    return float(sims.mean().item())


def mean_cosine_between_layers(h1: torch.Tensor, h2: torch.Tensor) -> float:
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    sims = (h1 * h2).sum(dim=1)
    return float(sims.mean().item())


def plot_tsne(emb, labels, out_path, title):
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
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================================================
# Dataset / subset
# =========================================================
def build_subset_data(full_data, subset_size=10000, seed=42):
    set_seed(seed)

    num_nodes = full_data.num_nodes
    perm = torch.randperm(num_nodes)[:subset_size]
    perm = perm.sort().values

    edge_index, _ = subgraph(
        subset=perm,
        edge_index=full_data.edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    edge_index = to_undirected(edge_index)

    x = full_data.x[perm].clone()
    y = full_data.y[perm].view(-1).clone()

    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    data = Data(x=x, edge_index=edge_index, y=y)

    n = subset_size
    order = torch.randperm(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def clone_data(data):
    return Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
        train_mask=data.train_mask.clone(),
        val_mask=data.val_mask.clone(),
        test_mask=data.test_mask.clone(),
    )


def add_feature_noise(x, noise_std):
    if noise_std == 0.0:
        return x.clone()
    noise = torch.randn_like(x) * noise_std
    return x + noise


def apply_edge_dropout(data, p):
    new_data = clone_data(data)
    if p == 0.0:
        return new_data
    new_edge_index, _ = dropout_edge(data.edge_index, p=p, force_undirected=True, training=True)
    new_data.edge_index = new_edge_index
    return new_data


def get_degree_buckets(data):
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    test_deg = deg[test_idx]

    q1 = np.percentile(test_deg, 33.33)
    q2 = np.percentile(test_deg, 66.66)

    low_idx = test_idx[test_deg <= q1]
    mid_idx = test_idx[(test_deg > q1) & (test_deg <= q2)]
    high_idx = test_idx[test_deg > q2]

    return {
        "thresholds": {"q33": float(q1), "q66": float(q2)},
        "low": low_idx,
        "medium": mid_idx,
        "high": high_idx,
    }


# =========================================================
# Training
# =========================================================
def train_one_run(model, data, lr, weight_decay, max_epochs=300, patience=50, min_delta=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    t0 = time.time()

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, val_f1 = evaluate(model, data, data.val_mask)

        if val_f1 > best_val_f1 + min_delta:
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

    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)

    return {
        "best_val_macro_f1": float(best_val_f1),
        "final_test_acc": float(final_test_acc),
        "final_test_macro_f1": float(final_test_f1),
        "train_time_sec": float(train_time),
    }


def build_baseline_model(model_name, in_channels, out_channels):
    if model_name == "GCN":
        return GCN2(in_channels, 128, out_channels, dropout=0.5, use_bn=True)
    if model_name == "GraphSAGE":
        return GraphSAGE2(in_channels, 128, out_channels, dropout=0.5, use_bn=True, aggr="mean")
    if model_name == "GAT":
        return GAT2(in_channels, 16, out_channels, heads=4, dropout=0.6)
    if model_name == "APPNP":
        return APPNPNet(in_channels, 128, out_channels, dropout=0.5, K=10, alpha=0.1)
    raise ValueError(model_name)


# =========================================================
# Experiments
# =========================================================
def run_baseline_all_models(data, num_features, num_classes, seeds, device):
    results = {}
    for model_name in ["GCN", "GraphSAGE", "GAT", "APPNP"]:
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            model = build_baseline_model(model_name, num_features, num_classes).to(device)
            metrics = train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
            metrics["seed"] = int(seed)
            per_seed.append(metrics)

        results[model_name] = {
            "num_parameters": int(count_parameters(build_baseline_model(model_name, num_features, num_classes))),
            "per_seed": per_seed,
            "summary": {
                "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                "train_time_sec_mean": mean_std([x["train_time_sec"] for x in per_seed])[0],
                "train_time_sec_std": mean_std([x["train_time_sec"] for x in per_seed])[1],
            }
        }
    return results


def run_depth_experiment(data, num_features, num_classes, seeds, device):
    cfg = {"hidden_dim": 128, "dropout": 0.5, "use_bn": True}
    records = []
    for model_name in ["GCN", "GraphSAGE"]:
        for depth in [2, 3, 4]:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                if model_name == "GCN":
                    model = GCNDepthEmb(num_features, cfg["hidden_dim"], num_classes, depth, cfg["dropout"], cfg["use_bn"]).to(device)
                else:
                    model = SAGEDepthEmb(num_features, cfg["hidden_dim"], num_classes, depth, cfg["dropout"], cfg["use_bn"], "mean").to(device)

                m = train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
                m["seed"] = int(seed)
                per_seed.append(m)

            records.append({
                "model": model_name,
                "depth": int(depth),
                "per_seed": per_seed,
                "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
            })
    return records


def run_graphsage_aggregator_experiment(data, num_features, num_classes, seeds, device):
    aggrs = ["mean", "max", "sum"]
    records = []
    for aggr in aggrs:
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            model = GraphSAGE2(num_features, 128, num_classes, dropout=0.5, use_bn=True, aggr=aggr).to(device)
            m = train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
            m["seed"] = int(seed)
            per_seed.append(m)

        records.append({
            "aggr": aggr,
            "per_seed": per_seed,
            "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
            "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
            "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
            "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
        })
    return records


def run_node_degree_experiment(data, num_features, num_classes, seeds, device):
    degree_buckets = get_degree_buckets(data)
    records = []
    for model_name in ["GCN", "GraphSAGE"]:
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            if model_name == "GCN":
                model = GCN2(num_features, 128, num_classes, dropout=0.5, use_bn=True).to(device)
            else:
                model = GraphSAGE2(num_features, 128, num_classes, dropout=0.5, use_bn=True, aggr="mean").to(device)

            m = train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
            low = evaluate_subset(model, data, degree_buckets["low"])
            mid = evaluate_subset(model, data, degree_buckets["medium"])
            high = evaluate_subset(model, data, degree_buckets["high"])

            per_seed.append({
                "seed": int(seed),
                "overall_test_acc": m["final_test_acc"],
                "overall_test_macro_f1": m["final_test_macro_f1"],
                "train_time_sec": m["train_time_sec"],
                "low_degree": low,
                "medium_degree": mid,
                "high_degree": high,
            })

        records.append({
            "model": model_name,
            "degree_thresholds": degree_buckets["thresholds"],
            "bucket_sizes": {
                "low": int(len(degree_buckets["low"])),
                "medium": int(len(degree_buckets["medium"])),
                "high": int(len(degree_buckets["high"])),
            },
            "per_seed": per_seed,
            "summary": {
                "overall_test_acc_mean": mean_std([x["overall_test_acc"] for x in per_seed])[0],
                "overall_test_acc_std": mean_std([x["overall_test_acc"] for x in per_seed])[1],
                "overall_test_macro_f1_mean": mean_std([x["overall_test_macro_f1"] for x in per_seed])[0],
                "overall_test_macro_f1_std": mean_std([x["overall_test_macro_f1"] for x in per_seed])[1],
                "low_degree_acc_mean": mean_std([x["low_degree"]["acc"] for x in per_seed])[0],
                "low_degree_acc_std": mean_std([x["low_degree"]["acc"] for x in per_seed])[1],
                "medium_degree_acc_mean": mean_std([x["medium_degree"]["acc"] for x in per_seed])[0],
                "medium_degree_acc_std": mean_std([x["medium_degree"]["acc"] for x in per_seed])[1],
                "high_degree_acc_mean": mean_std([x["high_degree"]["acc"] for x in per_seed])[0],
                "high_degree_acc_std": mean_std([x["high_degree"]["acc"] for x in per_seed])[1],
            }
        })
    return records


def run_feature_noise_experiment(data, num_features, num_classes, seeds, device):
    records = []
    for model_name in ["GCN", "GraphSAGE"]:
        for noise_std in [0.0, 0.1, 0.2, 0.3]:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                noisy = clone_data(data)
                noisy.x = add_feature_noise(data.x, noise_std)

                if model_name == "GCN":
                    model = GCN2(num_features, 128, num_classes, dropout=0.5, use_bn=True).to(device)
                else:
                    model = GraphSAGE2(num_features, 128, num_classes, dropout=0.5, use_bn=True, aggr="mean").to(device)

                m = train_one_run(model, noisy, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
                m["seed"] = int(seed)
                per_seed.append(m)

            records.append({
                "model": model_name,
                "noise_std": float(noise_std),
                "per_seed": per_seed,
                "summary": {
                    "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                    "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                    "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                    "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                }
            })
    return records


def run_edge_dropout_experiment(data, num_features, num_classes, seeds, device):
    records = []
    for model_name in ["GCN", "GraphSAGE"]:
        for drop_p in [0.0, 0.1, 0.2, 0.3]:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                dropped = apply_edge_dropout(data, drop_p)

                if model_name == "GCN":
                    model = GCN2(num_features, 128, num_classes, dropout=0.5, use_bn=True).to(device)
                else:
                    model = GraphSAGE2(num_features, 128, num_classes, dropout=0.5, use_bn=True, aggr="mean").to(device)

                m = train_one_run(model, dropped, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
                m["seed"] = int(seed)
                per_seed.append(m)

            records.append({
                "model": model_name,
                "edge_drop_p": float(drop_p),
                "per_seed": per_seed,
                "summary": {
                    "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                    "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                    "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                    "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                }
            })
    return records


def run_oversmoothing_experiment(data, num_features, num_classes, seeds, device):
    cfg = {"hidden_dim": 128, "dropout": 0.5, "use_bn": True}
    records = []
    for model_name in ["GCN", "GraphSAGE"]:
        for depth in [2, 3, 4]:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                if model_name == "GCN":
                    model = GCNDepthEmb(num_features, cfg["hidden_dim"], num_classes, depth, cfg["dropout"], cfg["use_bn"]).to(device)
                else:
                    model = SAGEDepthEmb(num_features, cfg["hidden_dim"], num_classes, depth, cfg["dropout"], cfg["use_bn"], "mean").to(device)

                train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
                model.eval()
                _, hidden = model(data.x, data.edge_index, return_hidden=True)

                sims = []
                for i, emb in enumerate(hidden, start=1):
                    sims.append({"hidden_layer": i, "avg_cosine_similarity": average_cosine_similarity(emb)})

                per_seed.append({"seed": int(seed), "layer_similarities": sims})

            num_hidden = len(per_seed[0]["layer_similarities"])
            summary = []
            for i in range(num_hidden):
                vals = [x["layer_similarities"][i]["avg_cosine_similarity"] for x in per_seed]
                summary.append({
                    "hidden_layer": i + 1,
                    "mean_avg_cosine_similarity": float(np.mean(vals)),
                    "std_avg_cosine_similarity": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                })

            records.append({
                "model": model_name,
                "depth": int(depth),
                "per_seed": per_seed,
                "summary": summary,
            })
    return records


def run_layer_rep_similarity_experiment(data, num_features, num_classes, seeds, device):
    cfg = {"hidden_dim": 128, "dropout": 0.5, "use_bn": True}
    records = []
    for model_name in ["GCN", "GraphSAGE"]:
        for depth in [3, 4]:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                if model_name == "GCN":
                    model = GCNDepthEmb(num_features, cfg["hidden_dim"], num_classes, depth, cfg["dropout"], cfg["use_bn"]).to(device)
                else:
                    model = SAGEDepthEmb(num_features, cfg["hidden_dim"], num_classes, depth, cfg["dropout"], cfg["use_bn"], "mean").to(device)

                train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
                model.eval()
                _, hidden = model(data.x, data.edge_index, return_hidden=True)

                pairwise = []
                for i in range(len(hidden) - 1):
                    pairwise.append({
                        "from_hidden_layer": i + 1,
                        "to_hidden_layer": i + 2,
                        "mean_cosine_similarity": mean_cosine_between_layers(hidden[i], hidden[i + 1]),
                    })

                per_seed.append({"seed": int(seed), "pairwise_layer_similarity": pairwise})

            num_pairs = len(per_seed[0]["pairwise_layer_similarity"])
            summary = []
            for i in range(num_pairs):
                vals = [x["pairwise_layer_similarity"][i]["mean_cosine_similarity"] for x in per_seed]
                summary.append({
                    "from_hidden_layer": per_seed[0]["pairwise_layer_similarity"][i]["from_hidden_layer"],
                    "to_hidden_layer": per_seed[0]["pairwise_layer_similarity"][i]["to_hidden_layer"],
                    "mean_cosine_similarity": float(np.mean(vals)),
                    "std_cosine_similarity": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                })

            records.append({
                "model": model_name,
                "depth": int(depth),
                "per_seed": per_seed,
                "summary": summary,
            })
    return records


def run_tsne_visualization(data, num_features, num_classes, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = [
        ("GCN", 2),
        ("GCN", 4),
        ("GraphSAGE", 2),
        ("GraphSAGE", 4),
    ]

    metadata = []
    for model_name, depth in runs:
        set_seed(42)
        if model_name == "GCN":
            model = GCNDepthEmb(num_features, 128, num_classes, depth, 0.5, True).to(device)
        else:
            model = SAGEDepthEmb(num_features, 128, num_classes, depth, 0.5, True, "mean").to(device)

        train_one_run(model, data, lr=0.01, weight_decay=5e-4, max_epochs=300, patience=50)
        model.eval()
        _, hidden = model(data.x, data.edge_index, return_hidden=True)

        last_hidden = hidden[-1].detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()

        run_name = f"ogbn_arxiv_subset_{model_name}_depth{depth}"
        png_path = os.path.join(out_dir, f"{run_name}.png")
        plot_tsne(last_hidden, labels, png_path, f"OGBN-Arxiv subset | {model_name} | depth={depth} | t-SNE")

        metadata.append({
            "model": model_name,
            "depth": int(depth),
            "file": png_path,
        })

    return metadata


# =========================================================
# Main
# =========================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = "experiments/ogbn_arxiv_subset_full_suite"
    os.makedirs(out_root, exist_ok=True)

    print("Loading ogbn-arxiv...")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
    full_data = dataset[0]

    print("Building 10k subset...")
    subset_data = build_subset_data(full_data, subset_size=10000, seed=42).to(device)

    num_features = subset_data.num_features
    num_classes = int(subset_data.y.max().item()) + 1

    seeds = [0, 1, 2]

    report = {
        "dataset": "ogbn-arxiv subset",
        "note": "All experiments are run on a sampled induced subgraph, not the full ogbn-arxiv dataset.",
        "subset_size": int(subset_data.num_nodes),
        "num_edges": int(subset_data.edge_index.size(1)),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
        "split": {
            "train": int(subset_data.train_mask.sum().item()),
            "val": int(subset_data.val_mask.sum().item()),
            "test": int(subset_data.test_mask.sum().item()),
        },
        "seeds": seeds,
    }

    print("\n[1/8] Running baseline all-model comparison...")
    report["baseline_all_models"] = run_baseline_all_models(subset_data, num_features, num_classes, seeds, device)

    print("\n[2/8] Running depth experiment...")
    report["depth_experiment"] = run_depth_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\n[3/8] Running GraphSAGE aggregator experiment...")
    report["graphsage_aggregator_experiment"] = run_graphsage_aggregator_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\n[4/8] Running node degree experiment...")
    report["node_degree_experiment"] = run_node_degree_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\n[5/8] Running feature noise experiment...")
    report["feature_noise_experiment"] = run_feature_noise_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\n[6/8] Running edge dropout experiment...")
    report["edge_dropout_experiment"] = run_edge_dropout_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\n[7/8] Running oversmoothing experiment...")
    report["oversmoothing_experiment"] = run_oversmoothing_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\n[8/8] Running layer representation similarity experiment...")
    report["layer_representation_similarity_experiment"] = run_layer_rep_similarity_experiment(subset_data, num_features, num_classes, seeds, device)

    print("\nGenerating t-SNE plots...")
    tsne_dir = os.path.join(out_root, "tsne_plots")
    report["tsne_visualization"] = run_tsne_visualization(subset_data, num_features, num_classes, device, tsne_dir)

    out_json = os.path.join(out_root, "results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved full suite results to: {out_json}")
    print(f"Saved t-SNE plots to: {tsne_dir}")


if __name__ == "__main__":
    main()
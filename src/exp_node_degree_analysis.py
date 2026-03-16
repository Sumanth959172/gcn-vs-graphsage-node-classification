import os
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import degree


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Models
# -----------------------------
class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6, use_bn=True):
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
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6, use_bn=True, aggr="mean"):
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


# -----------------------------
# Evaluation
# -----------------------------
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
        return {
            "num_nodes": 0,
            "acc": None,
            "macro_f1": None,
        }

    logits = model(data.x, data.edge_index)
    preds = logits[node_idx].argmax(dim=1).cpu().numpy()
    labels = data.y[node_idx].cpu().numpy()

    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))

    return {
        "num_nodes": int(len(node_idx)),
        "acc": acc,
        "macro_f1": macro_f1,
    }


# -----------------------------
# Training
# -----------------------------
def train_one_seed(model, data, lr, weight_decay, max_epochs, patience, min_delta=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(10, patience // 5), threshold=1e-4
    )

    best_val_f1 = -1.0
    best_state = None
    counter = 0

    t0 = time.time()

    for _epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, val_f1 = evaluate(model, data, data.val_mask)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    return float(best_val_f1), float(train_time)


# -----------------------------
# Utilities
# -----------------------------
def load_dataset(dataset_name, device):
    dataset = Planetoid(root=f"data/Planetoid/{dataset_name}", name=dataset_name)
    data = dataset[0]

    # Feature normalization
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    data = data.to(device)
    return dataset, data


def get_degree_buckets(data):
    """
    Split TEST nodes into low / medium / high degree buckets using
    33rd and 66th percentiles computed on test-node degrees.
    """
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()

    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    test_deg = deg[test_idx]

    q1 = np.percentile(test_deg, 33.33)
    q2 = np.percentile(test_deg, 66.66)

    low_idx = test_idx[test_deg <= q1]
    mid_idx = test_idx[(test_deg > q1) & (test_deg <= q2)]
    high_idx = test_idx[test_deg > q2]

    return {
        "thresholds": {
            "q33": float(q1),
            "q66": float(q2),
        },
        "low": low_idx,
        "medium": mid_idx,
        "high": high_idx,
    }


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) == 0:
        return None, None
    xs = np.array(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0, 1, 2, 3, 4]

    # Fixed configs for fair comparison
    cfg = {
        "hidden_dim": 256,
        "dropout": 0.6,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epochs": 1000,
        "patience": 200,
        "min_delta": 1e-4,
        "use_bn": True,
        "sage_aggr": "mean",
    }

    out_dir = os.path.join(
        "experiments",
        "node_degree_analysis",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for dataset_name in ["Cora", "PubMed"]:
        dataset, data = load_dataset(dataset_name, device)
        degree_buckets = get_degree_buckets(data)

        for model_name in ["GCN", "GraphSAGE"]:
            per_seed = []

            for seed in seeds:
                set_seed(seed)

                if model_name == "GCN":
                    model = GCN2(
                        in_channels=dataset.num_features,
                        hidden_channels=cfg["hidden_dim"],
                        out_channels=dataset.num_classes,
                        dropout=cfg["dropout"],
                        use_bn=cfg["use_bn"],
                    ).to(device)
                else:
                    model = GraphSAGE2(
                        in_channels=dataset.num_features,
                        hidden_channels=cfg["hidden_dim"],
                        out_channels=dataset.num_classes,
                        dropout=cfg["dropout"],
                        use_bn=cfg["use_bn"],
                        aggr=cfg["sage_aggr"],
                    ).to(device)

                best_val_f1, train_time = train_one_seed(
                    model=model,
                    data=data,
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    max_epochs=cfg["max_epochs"],
                    patience=cfg["patience"],
                    min_delta=cfg["min_delta"],
                )

                low_metrics = evaluate_subset(model, data, torch.tensor(degree_buckets["low"], device=device))
                mid_metrics = evaluate_subset(model, data, torch.tensor(degree_buckets["medium"], device=device))
                high_metrics = evaluate_subset(model, data, torch.tensor(degree_buckets["high"], device=device))

                overall_acc, overall_f1 = evaluate(model, data, data.test_mask)

                per_seed.append({
                    "seed": int(seed),
                    "best_val_macro_f1": float(best_val_f1),
                    "overall_test_acc": float(overall_acc),
                    "overall_test_macro_f1": float(overall_f1),
                    "train_time_sec": float(train_time),
                    "low_degree": low_metrics,
                    "medium_degree": mid_metrics,
                    "high_degree": high_metrics,
                })

            summary = {
                "overall_test_acc_mean": mean_std([x["overall_test_acc"] for x in per_seed])[0],
                "overall_test_acc_std": mean_std([x["overall_test_acc"] for x in per_seed])[1],
                "overall_test_macro_f1_mean": mean_std([x["overall_test_macro_f1"] for x in per_seed])[0],
                "overall_test_macro_f1_std": mean_std([x["overall_test_macro_f1"] for x in per_seed])[1],

                "low_degree_acc_mean": mean_std([x["low_degree"]["acc"] for x in per_seed])[0],
                "low_degree_acc_std": mean_std([x["low_degree"]["acc"] for x in per_seed])[1],
                "low_degree_f1_mean": mean_std([x["low_degree"]["macro_f1"] for x in per_seed])[0],
                "low_degree_f1_std": mean_std([x["low_degree"]["macro_f1"] for x in per_seed])[1],

                "medium_degree_acc_mean": mean_std([x["medium_degree"]["acc"] for x in per_seed])[0],
                "medium_degree_acc_std": mean_std([x["medium_degree"]["acc"] for x in per_seed])[1],
                "medium_degree_f1_mean": mean_std([x["medium_degree"]["macro_f1"] for x in per_seed])[0],
                "medium_degree_f1_std": mean_std([x["medium_degree"]["macro_f1"] for x in per_seed])[1],

                "high_degree_acc_mean": mean_std([x["high_degree"]["acc"] for x in per_seed])[0],
                "high_degree_acc_std": mean_std([x["high_degree"]["acc"] for x in per_seed])[1],
                "high_degree_f1_mean": mean_std([x["high_degree"]["macro_f1"] for x in per_seed])[0],
                "high_degree_f1_std": mean_std([x["high_degree"]["macro_f1"] for x in per_seed])[1],

                "train_time_sec_mean": mean_std([x["train_time_sec"] for x in per_seed])[0],
                "train_time_sec_std": mean_std([x["train_time_sec"] for x in per_seed])[1],
            }

            result = {
                "dataset": dataset_name,
                "model": model_name,
                "cfg": cfg,
                "degree_thresholds": degree_buckets["thresholds"],
                "bucket_sizes": {
                    "low": int(len(degree_buckets["low"])),
                    "medium": int(len(degree_buckets["medium"])),
                    "high": int(len(degree_buckets["high"])),
                },
                "seeds": seeds,
                "per_seed": per_seed,
                "summary": summary,
                "device": str(device),
            }

            all_results.append(result)

            print(f"\n[{dataset_name}] {model_name}")
            print(f"  overall acc={summary['overall_test_acc_mean']:.4f} ± {summary['overall_test_acc_std']:.4f}")
            print(f"  low-degree acc={summary['low_degree_acc_mean']:.4f} ± {summary['low_degree_acc_std']:.4f}")
            print(f"  medium-degree acc={summary['medium_degree_acc_mean']:.4f} ± {summary['medium_degree_acc_std']:.4f}")
            print(f"  high-degree acc={summary['high_degree_acc_mean']:.4f} ± {summary['high_degree_acc_std']:.4f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    meta = {
        "experiment": "node_degree_analysis",
        "datasets": ["Cora", "PubMed"],
        "models": ["GCN", "GraphSAGE"],
        "seeds": seeds,
        "cfg": cfg,
        "device": str(device),
        "saved_to": os.path.join(out_dir, "results.json"),
    }

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_dir}/results.json and meta.json")


if __name__ == "__main__":
    main()
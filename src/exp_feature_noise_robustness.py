
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


# -----------------------------
# Train
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

    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)

    return {
        "best_val_macro_f1": float(best_val_f1),
        "final_test_acc": float(final_test_acc),
        "final_test_macro_f1": float(final_test_f1),
        "train_time_sec": float(train_time),
    }


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


def add_feature_noise(x, noise_std):
    if noise_std == 0.0:
        return x.clone()
    noise = torch.randn_like(x) * noise_std
    return x + noise


def mean_std(xs):
    xs = np.array(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0, 1, 2, 3, 4]
    noise_levels = [0.0, 0.1, 0.2, 0.3]

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
        "feature_noise_robustness",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for dataset_name in ["Cora", "PubMed"]:
        dataset, clean_data = load_dataset(dataset_name, device)

        for model_name in ["GCN", "GraphSAGE"]:
            for noise_std in noise_levels:
                per_seed = []

                for seed in seeds:
                    set_seed(seed)

                    # clone data and corrupt features
                    data = clean_data.clone()
                    data.x = add_feature_noise(clean_data.x, noise_std)

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

                    metrics = train_one_seed(
                        model=model,
                        data=data,
                        lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"],
                        max_epochs=cfg["max_epochs"],
                        patience=cfg["patience"],
                        min_delta=cfg["min_delta"],
                    )
                    metrics["seed"] = int(seed)
                    per_seed.append(metrics)

                result = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "noise_std": float(noise_std),
                    "cfg": cfg,
                    "seeds": seeds,
                    "per_seed": per_seed,
                    "summary": {
                        "test_acc_mean": mean_std([x["final_test_acc"] for x in per_seed])[0],
                        "test_acc_std": mean_std([x["final_test_acc"] for x in per_seed])[1],
                        "test_macro_f1_mean": mean_std([x["final_test_macro_f1"] for x in per_seed])[0],
                        "test_macro_f1_std": mean_std([x["final_test_macro_f1"] for x in per_seed])[1],
                        "train_time_sec_mean": mean_std([x["train_time_sec"] for x in per_seed])[0],
                        "train_time_sec_std": mean_std([x["train_time_sec"] for x in per_seed])[1],
                    },
                    "device": str(device),
                }

                all_results.append(result)

                s = result["summary"]
                print(
                    f"[{dataset_name}] {model_name} noise={noise_std:.1f} | "
                    f"acc={s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f} | "
                    f"f1={s['test_macro_f1_mean']:.4f} ± {s['test_macro_f1_std']:.4f}"
                )

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    meta = {
        "experiment": "feature_noise_robustness",
        "datasets": ["Cora", "PubMed"],
        "models": ["GCN", "GraphSAGE"],
        "noise_levels": noise_levels,
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
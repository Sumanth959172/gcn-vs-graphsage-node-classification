import os
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import f1_score


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
# Depth-controlled models
# -----------------------------
class GCNDepth(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.5, use_bn=True):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # first layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if self.use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # hidden layers
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        # output layer
        self.out_conv = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_conv(x, edge_index)
        return x


class SAGEDepth(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, dropout=0.5, use_bn=True, aggr="mean"):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.aggr = aggr

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

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_conv(x, edge_index)
        return x


# -----------------------------
# Train (early stop by val macro-F1)
# -----------------------------
def train_one_seed(
    model,
    data,
    lr,
    weight_decay,
    max_epochs,
    patience,
    min_delta=1e-4,
):
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

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    val_acc, val_f1 = evaluate(model, data, data.val_mask)
    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    return {
        "best_val_macro_f1": float(best_val_f1),
        "final_val_acc": float(val_acc),
        "final_val_macro_f1": float(val_f1),
        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(test_f1),
        "train_time_sec": float(train_time),
    }


def mean_std(xs):
    xs = np.array(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def run_depth_sweep(dataset_name, model_name, depths, dataset, data, device, seeds, cfg):
    records = []

    for depth in depths:
        per_seed = []
        for seed in seeds:
            set_seed(seed)

            if model_name == "GCN":
                model = GCNDepth(
                    in_channels=dataset.num_features,
                    hidden_channels=cfg["hidden_dim"],
                    out_channels=dataset.num_classes,
                    depth=depth,
                    dropout=cfg["dropout"],
                    use_bn=cfg["use_bn"],
                ).to(device)
            elif model_name == "GraphSAGE":
                model = SAGEDepth(
                    in_channels=dataset.num_features,
                    hidden_channels=cfg["hidden_dim"],
                    out_channels=dataset.num_classes,
                    depth=depth,
                    dropout=cfg["dropout"],
                    use_bn=cfg["use_bn"],
                    aggr=cfg["sage_aggr"],
                ).to(device)
            else:
                raise ValueError("model_name must be 'GCN' or 'GraphSAGE'")

            m = train_one_seed(
                model=model,
                data=data,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                min_delta=cfg["min_delta"],
            )
            m["seed"] = int(seed)
            per_seed.append(m)

        test_acc_mean, test_acc_std = mean_std([m["final_test_acc"] for m in per_seed])
        test_f1_mean, test_f1_std = mean_std([m["final_test_macro_f1"] for m in per_seed])
        val_f1_mean, val_f1_std = mean_std([m["best_val_macro_f1"] for m in per_seed])
        time_mean, time_std = mean_std([m["train_time_sec"] for m in per_seed])

        record = {
            "dataset": dataset_name,
            "model": model_name,
            "depth": int(depth),
            "cfg": cfg,
            "seeds": list(map(int, seeds)),
            "per_seed": per_seed,
            "best_val_macro_f1_mean": val_f1_mean,
            "best_val_macro_f1_std": val_f1_std,
            "test_acc_mean": test_acc_mean,
            "test_acc_std": test_acc_std,
            "test_macro_f1_mean": test_f1_mean,
            "test_macro_f1_std": test_f1_std,
            "train_time_sec_mean": time_mean,
            "train_time_sec_std": time_std,
            "device": str(device),
        }
        records.append(record)

        print(
            f"[{dataset_name}] {model_name} depth={depth} | "
            f"test_acc={test_acc_mean:.4f}±{test_acc_std:.4f} | "
            f"test_f1={test_f1_mean:.4f}±{test_f1_std:.4f}"
        )

    return records


def load_dataset(name, device):
    dataset = Planetoid(root=f"data/Planetoid/{name}", name=name)
    data = dataset[0]

    # Feature normalization (consistent)
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    data = data.to(device)
    return dataset, data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0, 1, 2, 3, 4]
    depths = [2, 3, 4]

    # Keep this fixed so depth is the only variable
    cfg = {
        "hidden_dim": 256,     # strong default for both models
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
        "depth_gcn_vs_graphsage",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for dataset_name in ["Cora", "PubMed"]:
        dataset, data = load_dataset(dataset_name, device)

        for model_name in ["GCN", "GraphSAGE"]:
            records = run_depth_sweep(
                dataset_name=dataset_name,
                model_name=model_name,
                depths=depths,
                dataset=dataset,
                data=data,
                device=device,
                seeds=seeds,
                cfg=cfg,
            )
            all_results.extend(records)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    meta = {
        "experiment": "depth_sweep_gcn_vs_graphsage",
        "datasets": ["Cora", "PubMed"],
        "models": ["GCN", "GraphSAGE"],
        "depths": depths,
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
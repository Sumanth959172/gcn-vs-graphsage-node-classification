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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_cosine_between_layers(h1: torch.Tensor, h2: torch.Tensor) -> float:
    """
    Mean cosine similarity between corresponding node representations
    in two consecutive hidden layers.
    """
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    sims = (h1 * h2).sum(dim=1)
    return float(sims.mean().item())


class GCNDepthRep(nn.Module):
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

    def forward(self, x, edge_index, return_hidden=False):
        hidden_list = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            hidden_list.append(x.detach())
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.out_conv(x, edge_index)

        if return_hidden:
            return out, hidden_list
        return out


class SAGEDepthRep(nn.Module):
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

    def forward(self, x, edge_index, return_hidden=False):
        hidden_list = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            hidden_list.append(x.detach())
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.out_conv(x, edge_index)

        if return_hidden:
            return out, hidden_list
        return out


@torch.no_grad()
def evaluate_acc(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = logits[mask].argmax(dim=1)
    labels = data.y[mask]
    return float((preds == labels).float().mean().item())


def train_one_seed(model, data, lr, weight_decay, max_epochs, patience, min_delta=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    counter = 0

    t0 = time.time()

    for _ in range(max_epochs):
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

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    return float(train_time), float(best_val_acc)


def load_dataset(name, device):
    dataset = Planetoid(root=f"data/Planetoid/{name}", name=name)
    data = dataset[0]

    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    return dataset, data.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = ["Cora", "PubMed"]
    depths = [3, 4]   # need at least 2 hidden layers to compare consecutive layers
    seeds = [0, 1, 2]

    cfg = {
        "hidden_dim": 256,
        "dropout": 0.6,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "max_epochs": 500,
        "patience": 100,
        "use_bn": True,
        "sage_aggr": "mean",
    }

    out_dir = os.path.join(
        "experiments",
        "layer_rep_similarity",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for dataset_name in datasets:
        dataset, data = load_dataset(dataset_name, device)

        for model_name in ["GCN", "GraphSAGE"]:
            for depth in depths:
                per_seed = []

                for seed in seeds:
                    set_seed(seed)

                    if model_name == "GCN":
                        model = GCNDepthRep(
                            dataset.num_features,
                            cfg["hidden_dim"],
                            dataset.num_classes,
                            depth=depth,
                            dropout=cfg["dropout"],
                            use_bn=cfg["use_bn"],
                        ).to(device)
                    else:
                        model = SAGEDepthRep(
                            dataset.num_features,
                            cfg["hidden_dim"],
                            dataset.num_classes,
                            depth=depth,
                            dropout=cfg["dropout"],
                            use_bn=cfg["use_bn"],
                            aggr=cfg["sage_aggr"],
                        ).to(device)

                    train_time, best_val_acc = train_one_seed(
                        model=model,
                        data=data,
                        lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"],
                        max_epochs=cfg["max_epochs"],
                        patience=cfg["patience"],
                    )

                    model.eval()
                    _, hidden_list = model(data.x, data.edge_index, return_hidden=True)

                    pairwise = []
                    for i in range(len(hidden_list) - 1):
                        sim = mean_cosine_between_layers(hidden_list[i], hidden_list[i + 1])
                        pairwise.append({
                            "from_hidden_layer": i + 1,
                            "to_hidden_layer": i + 2,
                            "mean_cosine_similarity": sim,
                        })

                    per_seed.append({
                        "seed": int(seed),
                        "best_val_acc": float(best_val_acc),
                        "train_time_sec": float(train_time),
                        "pairwise_layer_similarity": pairwise,
                    })

                # aggregate
                num_pairs = len(per_seed[0]["pairwise_layer_similarity"])
                summary = []

                for i in range(num_pairs):
                    vals = [seed_res["pairwise_layer_similarity"][i]["mean_cosine_similarity"] for seed_res in per_seed]
                    summary.append({
                        "from_hidden_layer": per_seed[0]["pairwise_layer_similarity"][i]["from_hidden_layer"],
                        "to_hidden_layer": per_seed[0]["pairwise_layer_similarity"][i]["to_hidden_layer"],
                        "mean_cosine_similarity": float(np.mean(vals)),
                        "std_cosine_similarity": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    })

                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "depth": int(depth),
                    "cfg": cfg,
                    "seeds": seeds,
                    "per_seed": per_seed,
                    "summary": summary,
                }
                results.append(record)

                print(f"\n[{dataset_name}] {model_name} depth={depth}")
                for row in summary:
                    print(
                        f"  hidden {row['from_hidden_layer']} -> {row['to_hidden_layer']} | "
                        f"cos_sim={row['mean_cosine_similarity']:.4f} ± {row['std_cosine_similarity']:.4f}"
                    )

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
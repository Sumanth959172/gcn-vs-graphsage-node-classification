import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
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


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")
    return float(acc), float(macro_f1)


def train_one_seed(data, in_ch, out_ch, cfg, seed, device):
    set_seed(seed)

    model = GAT(
        in_ch,
        cfg["hidden_dim"],
        out_ch,
        cfg["heads"],
        cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=30
    )

    best_val_f1 = -1.0
    best_state = None
    patience = 200
    counter = 0
    max_epochs = 2000
    min_delta = 1e-4

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

    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    return {
        "seed": seed,
        "best_val_macro_f1": float(best_val_f1),
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "train_time_sec": float(train_time),
    }


def main():
    cfg = {
        "hidden_dim": 8,
        "heads": 4,
        "dropout": 0.6,
        "lr": 0.02,
        "weight_decay": 0.001,
    }

    dataset = Planetoid(root="data/Planetoid/Cora", name="Cora")
    data = dataset[0]

    # Feature normalization
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    seeds = [0, 1, 2, 3, 4]

    results = []
    for seed in seeds:
        r = train_one_seed(data, dataset.num_features, dataset.num_classes, cfg, seed, device)
        results.append(r)
        print(f"seed={seed} | acc={r['test_acc']:.4f} | f1={r['test_macro_f1']:.4f}")

    accs = [r["test_acc"] for r in results]
    f1s = [r["test_macro_f1"] for r in results]
    times = [r["train_time_sec"] for r in results]

    summary = {
        "model": "GAT",
        "dataset": "Cora",
        "selection_metric": "val_macro_f1",
        "cfg": cfg,
        "test_acc_mean": float(np.mean(accs)),
        "test_acc_std": float(np.std(accs, ddof=1)),
        "test_macro_f1_mean": float(np.mean(f1s)),
        "test_macro_f1_std": float(np.std(f1s, ddof=1)),
        "train_time_sec_mean": float(np.mean(times)),
        "train_time_sec_std": float(np.std(times, ddof=1)),
        "per_seed": results,
        "device": str(device),
    }

    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))

    out_dir = "experiments/final_cora_gat"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {os.path.join(out_dir, 'final.json')}")


if __name__ == "__main__":
    main()
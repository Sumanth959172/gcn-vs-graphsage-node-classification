import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCN3(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
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


def train_one_seed(
    data,
    in_channels,
    out_channels,
    hidden_dim,
    dropout,
    lr,
    weight_decay,
    max_epochs,
    seed,
    device,
    patience=200,
    min_delta=1e-4,
):
    set_seed(seed)

    model = GCN3(in_channels, hidden_dim, out_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=30, threshold=1e-4
    )

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

        val_acc, val_f1 = evaluate(model, data, data.val_mask)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    train_time = time.time() - t0

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    final_val_acc, final_val_f1 = evaluate(model, data, data.val_mask)
    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)

    return {
        "seed": int(seed),
        "best_val_macro_f1": float(best_val_f1),
        "final_val_acc": float(final_val_acc),
        "final_val_macro_f1": float(final_val_f1),
        "final_test_acc": float(final_test_acc),
        "final_test_macro_f1": float(final_test_f1),
        "train_time_sec": float(train_time),
    }


def mean_std(xs):
    xs = np.array(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def main():
    # --- Use your tuned hyperparameters (from best.json you pasted) ---
    cfg = {
        "hidden_dim": 128,
        "dropout": 0.7,
        "lr": 0.02,
        "weight_decay": 5e-4,
    }

    dataset_name = "Cora"
    root = f"data/Planetoid/{dataset_name}"

    # Final seeds (5-seed reporting)
    seeds = [0, 1, 2, 3, 4]

    # Cora settings (match tuning behavior)
    max_epochs = 2000
    patience = 200
    min_delta = 1e-4

    out_dir = "experiments/final_cora_gcn3"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]

    # Feature normalization (keep consistent)
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    data = data.to(device)

    print("FINAL EVAL Cora GCN3")
    print("Using cfg:", cfg)
    print("Device:", device)
    print("Seeds:", seeds)

    per_seed = []
    for seed in seeds:
        m = train_one_seed(
            data=data,
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            max_epochs=max_epochs,
            seed=seed,
            device=device,
            patience=patience,
            min_delta=min_delta,
        )
        per_seed.append(m)
        print(f"seed={seed} | test_acc={m['final_test_acc']:.4f} | test_f1={m['final_test_macro_f1']:.4f}")

    test_accs = [m["final_test_acc"] for m in per_seed]
    test_f1s = [m["final_test_macro_f1"] for m in per_seed]
    times = [m["train_time_sec"] for m in per_seed]

    acc_mean, acc_std = mean_std(test_accs)
    f1_mean, f1_std = mean_std(test_f1s)
    t_mean, t_std = mean_std(times)

    summary = {
        "model": "GCN3 (BN)",
        "dataset": dataset_name,
        "selection_metric": "val_macro_f1",
        "cfg": cfg,
        "seeds": seeds,
        "per_seed": per_seed,
        "test_acc_mean": acc_mean,
        "test_acc_std": acc_std,
        "test_macro_f1_mean": f1_mean,
        "test_macro_f1_std": f1_std,
        "train_time_sec_mean": t_mean,
        "train_time_sec_std": t_std,
        "device": str(device),
    }

    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {os.path.join(out_dir, 'final.json')}")


if __name__ == "__main__":
    main()
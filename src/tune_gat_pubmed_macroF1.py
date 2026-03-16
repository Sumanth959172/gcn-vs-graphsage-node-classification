import os
import json
import time
import random
import itertools
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


# -----------------------------
# 2-layer GAT
# -----------------------------
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

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


def train_one_run(
    data,
    in_channels,
    out_channels,
    hidden_dim,
    heads,
    dropout,
    lr,
    weight_decay,
    max_epochs,
    seed,
    device,
    patience=100,
    min_delta=1e-4,
):
    set_seed(seed)

    model = GAT(
        in_channels,
        hidden_dim,
        out_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, threshold=1e-4
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
        "seed": seed,
        "best_val_macro_f1": float(best_val_f1),
        "final_test_acc": float(final_test_acc),
        "final_test_macro_f1": float(final_test_f1),
        "train_time_sec": float(train_time),
    }


def main():
    dataset_name = "PubMed"
    root = f"data/Planetoid/{dataset_name}"

    max_epochs = 1000
    patience = 100

    random_trials = 30
    seeds = [0, 1, 2]

    space = {
        "hidden_dim": [8, 16, 32],
        "heads": [4, 8],
        "dropout": [0.5, 0.6, 0.7],
        "lr": [0.001, 0.003, 0.005],
        "weight_decay": [0.0, 5e-4, 1e-3],
    }

    out_dir = "experiments/tune_pubmed_gat_macroF1"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]

    # Feature normalization
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    data = data.to(device)

    all_cfgs = list(itertools.product(
        space["hidden_dim"],
        space["heads"],
        space["dropout"],
        space["lr"],
        space["weight_decay"],
    ))

    random.seed(12345)
    all_cfgs = random.sample(all_cfgs, k=min(random_trials, len(all_cfgs)))

    trials_log = []
    best_trial = None

    for trial_idx, (hidden_dim, heads, dropout, lr, wd) in enumerate(all_cfgs, start=1):

        per_seed = []

        for seed in seeds:
            m = train_one_run(
                data,
                dataset.num_features,
                dataset.num_classes,
                hidden_dim,
                heads,
                dropout,
                lr,
                wd,
                max_epochs,
                seed,
                device,
                patience,
            )
            per_seed.append(m)

        mean_val_f1 = float(np.mean([m["best_val_macro_f1"] for m in per_seed]))

        trial_record = {
            "trial": trial_idx,
            "cfg": {
                "hidden_dim": hidden_dim,
                "heads": heads,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": wd,
            },
            "mean_best_val_macro_f1": mean_val_f1,
            "per_seed": per_seed,
        }

        trials_log.append(trial_record)

        if (best_trial is None) or (mean_val_f1 > best_trial["mean_best_val_macro_f1"]):
            best_trial = trial_record
            print(f"✅ NEW BEST: {mean_val_f1:.4f}")

        with open(os.path.join(out_dir, "trials.json"), "w") as f:
            json.dump(trials_log, f, indent=2)

        with open(os.path.join(out_dir, "best.json"), "w") as f:
            json.dump(best_trial, f, indent=2)

    print("\nBEST CONFIG:")
    print(json.dumps(best_trial, indent=2))


if __name__ == "__main__":
    main()
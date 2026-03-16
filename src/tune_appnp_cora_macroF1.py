
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
from torch_geometric.nn import APPNP
from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP_APPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, K, alpha):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = float(dropout)
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
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
    dropout,
    K,
    alpha,
    lr,
    weight_decay,
    max_epochs,
    seed,
    device,
    patience=200,
    min_delta=1e-4,
):
    set_seed(seed)

    model = MLP_APPNP(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        dropout=dropout,
        K=K,
        alpha=alpha,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=30, threshold=1e-4
    )

    best_val_f1 = -1.0
    best_state = None
    counter = 0

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
        "seed": int(seed),
        "best_val_macro_f1": float(best_val_f1),
        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(test_f1),
        "train_time_sec": float(train_time),
    }


def main():
    dataset_name = "Cora"
    root = f"data/Planetoid/{dataset_name}"

    # Cora: longer training + more patience (val is noisy)
    max_epochs = 2000
    patience = 200
    min_delta = 1e-4

    random_trials = 30
    seeds = [0, 1, 2]

    # Cora-optimized APPNP space
    # Cora usually likes K=10..20 and alpha=0.1..0.2, smaller hidden dims often work well.
    space = {
        "hidden_dim": [64, 128, 256],
        "dropout": [0.3, 0.5, 0.6, 0.7],
        "K": [10, 20],
        "alpha": [0.1, 0.2],
        "lr": [0.005, 0.01, 0.02],
        "weight_decay": [0.0, 5e-4, 1e-3],
    }

    out_dir = "experiments/tune_cora_appnp_macroF1"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]

    # Feature normalization (consistent)
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    data = data.to(device)

    all_cfgs = list(itertools.product(
        space["hidden_dim"],
        space["dropout"],
        space["K"],
        space["alpha"],
        space["lr"],
        space["weight_decay"],
    ))

    random.seed(12345)
    all_cfgs = random.sample(all_cfgs, k=min(random_trials, len(all_cfgs)))

    print(f"Device: {device}")
    print(f"Model: MLP+APPNP | Dataset: {dataset_name}")
    print(f"Trials: {len(all_cfgs)} | Seeds/trial: {len(seeds)} | Total runs: {len(all_cfgs) * len(seeds)}")
    print(f"Saving to: {out_dir}")

    trials_log = []
    best_trial = None

    for trial_idx, (hidden_dim, dropout, K, alpha, lr, wd) in enumerate(all_cfgs, start=1):
        cfg = {
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "K": int(K),
            "alpha": float(alpha),
            "lr": float(lr),
            "weight_decay": float(wd),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "min_delta": float(min_delta),
        }

        print("\n" + "=" * 90)
        print(f"TRIAL {trial_idx}/{len(all_cfgs)} | cfg={cfg}")

        per_seed = []
        for seed in seeds:
            m = train_one_run(
                data=data,
                in_channels=dataset.num_features,
                out_channels=dataset.num_classes,
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
                K=cfg["K"],
                alpha=cfg["alpha"],
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                max_epochs=cfg["max_epochs"],
                seed=seed,
                device=device,
                patience=cfg["patience"],
                min_delta=cfg["min_delta"],
            )
            per_seed.append(m)

        mean_val_f1 = float(np.mean([m["best_val_macro_f1"] for m in per_seed]))
        mean_test_acc = float(np.mean([m["final_test_acc"] for m in per_seed]))
        mean_test_f1 = float(np.mean([m["final_test_macro_f1"] for m in per_seed]))
        mean_time = float(np.mean([m["train_time_sec"] for m in per_seed]))

        trial_record = {
            "trial": int(trial_idx),
            "cfg": cfg,
            "seeds": list(map(int, seeds)),
            "per_seed": per_seed,
            "mean_best_val_macro_f1": mean_val_f1,
            "mean_final_test_acc": mean_test_acc,
            "mean_final_test_macro_f1": mean_test_f1,
            "mean_train_time_sec": mean_time,
            "device": str(device),
        }
        trials_log.append(trial_record)

        if (best_trial is None) or (mean_val_f1 > best_trial["mean_best_val_macro_f1"]):
            best_trial = trial_record
            print(f"✅ NEW BEST by mean val macro-F1: {mean_val_f1:.4f}")

        with open(os.path.join(out_dir, "trials.json"), "w") as f:
            json.dump(trials_log, f, indent=2)

        with open(os.path.join(out_dir, "best.json"), "w") as f:
            json.dump(best_trial, f, indent=2)

        print(
            f"Trial summary | mean_val_f1={mean_val_f1:.4f} | "
            f"mean_test_acc={mean_test_acc:.4f} | mean_test_f1={mean_test_f1:.4f} | "
            f"mean_time={mean_time:.1f}s"
        )

    print("\nBEST CONFIG (selected by mean val macro-F1):")
    print(json.dumps(best_trial, indent=2))


if __name__ == "__main__":
    main()
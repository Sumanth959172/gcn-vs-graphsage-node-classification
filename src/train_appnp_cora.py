import os
import json
import time
import random
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
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return acc, macro_f1


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="data/Planetoid/Cora", name="Cora")
    data = dataset[0]

    # Feature normalization
    x = data.x
    data.x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
    data = data.to(device)

    # Baseline hyperparams (reasonable default)
    cfg = {
        "hidden_dim": 128,
        "dropout": 0.7,
        "K": 10,
        "alpha": 0.1,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 2000,
    }

    model = MLP_APPNP(
        in_channels=dataset.num_features,
        hidden_channels=cfg["hidden_dim"],
        out_channels=dataset.num_classes,
        dropout=cfg["dropout"],
        K=cfg["K"],
        alpha=cfg["alpha"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "test_acc": [],
        "test_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_state = None

    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        val_acc, val_f1 = evaluate(model, data, data.val_mask)
        test_acc, test_f1 = evaluate(model, data, data.test_mask)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["test_acc"].append(test_acc)
        history["test_macro_f1"].append(test_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:04d} | loss={loss.item():.4f} | "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
            )

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_acc, final_val_f1 = evaluate(model, data, data.val_mask)
    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)

    results = {
        "model": "MLP+APPNP",
        "dataset": "Cora",
        "selection_metric": "val_macro_f1",
        "cfg": cfg,
        "best_val_macro_f1": float(best_val_f1),
        "final_val_acc": float(final_val_acc),
        "final_val_macro_f1": float(final_val_f1),
        "final_test_acc": float(final_test_acc),
        "final_test_macro_f1": float(final_test_f1),
        "device": str(device),
        "total_train_time_sec": float(train_time),
        "seed": 42,
    }

    print("\nFINAL (best by val macro-F1)")
    print(json.dumps(results, indent=2))

    out_dir = "experiments/train_cora_appnp"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to: {out_dir}/results.json and history.json")


if __name__ == "__main__":
    main()
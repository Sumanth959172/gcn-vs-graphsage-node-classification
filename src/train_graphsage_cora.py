import json
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
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


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root="data/Planetoid/Cora", name="Cora")
    data = dataset[0].to(device)

    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_f1 = -1.0
    best_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "test_acc": [],
        "test_macro_f1": [],
    }

    start_time = time.time()

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
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
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Val Acc {val_acc:.4f} F1 {val_f1:.4f} | "
                f"Test Acc {test_acc:.4f} F1 {test_f1:.4f}"
            )

    total_time = time.time() - start_time

    # Load best model by validation macro-F1
    model.load_state_dict(best_state)

    final_val_acc, final_val_f1 = evaluate(model, data, data.val_mask)
    final_test_acc, final_test_f1 = evaluate(model, data, data.test_mask)

    results = {
        "model": "GraphSAGE(mean)",
        "dataset": "Cora",
        "hidden_dim": 64,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 200,
        "best_val_macro_f1": best_val_f1,
        "final_val_acc": final_val_acc,
        "final_val_macro_f1": final_val_f1,
        "final_test_acc": final_test_acc,
        "final_test_macro_f1": final_test_f1,
        "device": str(device),
        "total_train_time_sec": total_time,
    }

    print("\nFINAL (best by val macro-F1)")
    print(json.dumps(results, indent=2))

    out_dir = "experiments/exp02_cora_graphsage"
    import os
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to: {out_dir}/results.json and history.json")


if __name__ == "__main__":
    main()

import os
import json
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP


# -----------------------------
# Model definitions (for parameter counting)
# -----------------------------
class GCN3(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        raise NotImplementedError


class GraphSAGE2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        raise NotImplementedError


class GAT2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        raise NotImplementedError


class MLPAPPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6, K=20, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
        self.dropout = float(dropout)

    def forward(self, x, edge_index):
        raise NotImplementedError


# -----------------------------
# Helpers
# -----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_model(dataset_name, model_name):
    # dataset dimensions
    dims = {
        "Cora": {"in": 1433, "out": 7},
        "PubMed": {"in": 500, "out": 3},
    }

    in_channels = dims[dataset_name]["in"]
    out_channels = dims[dataset_name]["out"]

    if model_name == "GCN3":
        return GCN3(in_channels, 128, out_channels, dropout=0.7)

    if model_name == "GraphSAGE":
        return GraphSAGE2(in_channels, 256, out_channels, dropout=0.6, aggr="mean")

    if model_name == "GAT":
        if dataset_name == "Cora":
            return GAT2(in_channels, 8, out_channels, heads=4, dropout=0.6)
        else:
            return GAT2(in_channels, 16, out_channels, heads=4, dropout=0.6)

    if model_name == "APPNP":
        if dataset_name == "Cora":
            return MLPAPPNP(in_channels, 128, out_channels, dropout=0.7, K=20, alpha=0.1)
        else:
            return MLPAPPNP(in_channels, 256, out_channels, dropout=0.6, K=20, alpha=0.1)

    raise ValueError(f"Unknown model: {model_name}")


def main():
    # Paths to your final experiment files
    files = {
        ("Cora", "GCN3"): "experiments/final_cora_gcn3/final.json",
        ("Cora", "GraphSAGE"): "experiments/final_cora_graphsage/final.json",
        ("Cora", "GAT"): "experiments/final_cora_gat/final.json",
        ("Cora", "APPNP"): "experiments/final_cora_appnp/final.json",

        ("PubMed", "GCN3"): "experiments/final_pubmed_gcn3/final.json",
        ("PubMed", "GraphSAGE"): "experiments/final_pubmed_graphsage/final.json",
        ("PubMed", "GAT"): "experiments/final_pubmed_gat/final.json",
        ("PubMed", "APPNP"): "experiments/final_pubmed_appnp/final.json",
    }

    rows = []

    for (dataset_name, model_name), path in files.items():
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        result = load_json(path)
        model = build_model(dataset_name, model_name)
        num_params = count_parameters(model)

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "num_parameters": int(num_params),
            "test_acc_mean": float(result["test_acc_mean"]),
            "test_acc_std": float(result["test_acc_std"]),
            "test_macro_f1_mean": float(result["test_macro_f1_mean"]),
            "test_macro_f1_std": float(result["test_macro_f1_std"]),
            "train_time_sec_mean": float(result["train_time_sec_mean"]) if "train_time_sec_mean" in result else None,
            "train_time_sec_std": float(result["train_time_sec_std"]) if "train_time_sec_std" in result else None,
        }
        rows.append(row)

    out_dir = os.path.join("experiments", "parameter_efficiency")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nPARAMETER EFFICIENCY SUMMARY\n")
    for dataset_name in ["Cora", "PubMed"]:
        print(dataset_name)
        print("-" * 95)
        print(
            f"{'Model':<15} {'Params':<12} {'Acc':<20} {'Macro-F1':<20} {'Train Time (s)':<20}"
        )
        for row in rows:
            if row["dataset"] == dataset_name:
                acc_str = f"{row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}"
                f1_str = f"{row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_std']:.4f}"
                time_str = (
                    f"{row['train_time_sec_mean']:.2f} ± {row['train_time_sec_std']:.2f}"
                    if row["train_time_sec_mean"] is not None else "N/A"
                )
                print(
                    f"{row['model']:<15} {row['num_parameters']:<12} {acc_str:<20} {f1_str:<20} {time_str:<20}"
                )
        print()

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
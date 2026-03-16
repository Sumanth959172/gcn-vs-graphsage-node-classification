"""
Dataset stats script (GCN vs GraphSAGE case study)

- Loads: Cora, PubMed (Planetoid), ogbn-arxiv (OGB)
- Prints a clean comparison table
- Exports: dataset_overview.md
"""

import torch
import pandas as pd
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


# ---------------------------------------------------------------
# PyTorch 2.6+ compatibility for OGB processed PyG files
# ---------------------------------------------------------------
_torch_load = torch.load


def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = patched_load
# ---------------------------------------------------------------


def planetoid_stats(name: str) -> dict:
    dataset = Planetoid(root=f"data/Planetoid/{name}", name=name)
    data = dataset[0]

    return {
        "dataset": name,
        "family": "Planetoid",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
    }


def ogbn_stats(name: str) -> dict:
    dataset = PygNodePropPredDataset(name=name, root="data")
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    num_classes = int(dataset.num_classes)
    if hasattr(num_classes, "item"):
        num_classes = int(num_classes.item())

    return {
        "dataset": name,
        "family": "OGB",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(dataset.num_features),
        "num_classes": num_classes,
        "train_nodes": int(train_idx.numel()),
        "val_nodes": int(valid_idx.numel()),
        "test_nodes": int(test_idx.numel()),
    }


def main():
    rows = [
        planetoid_stats("Cora"),
        planetoid_stats("PubMed"),
        ogbn_stats("ogbn-arxiv"),
    ]

    df = pd.DataFrame(rows)

    df = df[
        [
            "dataset",
            "family",
            "num_nodes",
            "num_edges",
            "num_features",
            "num_classes",
            "train_nodes",
            "val_nodes",
            "test_nodes",
        ]
    ].sort_values(by="num_nodes")

    print(df.to_string(index=False))

    # Export markdown table
    md = df.to_markdown(index=False)
    with open("dataset_overview.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Overview\n\n")
        f.write(md)
        f.write("\n")

    print("\nSaved: dataset_overview.md")


if __name__ == "__main__":
    main()
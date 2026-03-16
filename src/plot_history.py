import json
import os
import matplotlib.pyplot as plt


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_curves(history, title, out_path_prefix):
    epochs = history["epoch"]

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"{title} - Training Loss")
    plt.savefig(out_path_prefix + "_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Macro-F1
    plt.figure()
    plt.plot(epochs, history["val_macro_f1"], label="Val Macro-F1")
    plt.plot(epochs, history["test_macro_f1"], label="Test Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{title} - Macro-F1")
    plt.legend()
    plt.savefig(out_path_prefix + "_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    gcn_hist = load_history("experiments/exp01_cora_gcn/history.json")
    sage_hist = load_history("experiments/exp02_cora_graphsage/history.json")

    os.makedirs("experiments/plots", exist_ok=True)

    plot_curves(gcn_hist, "GCN (Cora)", "experiments/plots/gcn_cora")
    plot_curves(sage_hist, "GraphSAGE (Cora)", "experiments/plots/graphsage_cora")

    print("Saved plots in experiments/plots/")


if __name__ == "__main__":
    main()

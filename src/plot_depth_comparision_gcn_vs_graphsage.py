import os
import json
import math
import argparse
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def organize_results(results):
    """
    Organize as:
    organized[dataset][model] = {
        "depths": [...],
        "acc_mean": [...],
        "acc_std": [...],
        "f1_mean": [...],
        "f1_std": [...]
    }
    """
    organized = {}

    for row in results:
        dataset = row["dataset"]
        model = row["model"]
        depth = row["depth"]

        organized.setdefault(dataset, {})
        organized[dataset].setdefault(model, {
            "depths": [],
            "acc_mean": [],
            "acc_std": [],
            "f1_mean": [],
            "f1_std": [],
        })

        organized[dataset][model]["depths"].append(depth)
        organized[dataset][model]["acc_mean"].append(row["test_acc_mean"])
        organized[dataset][model]["acc_std"].append(row["test_acc_std"])
        organized[dataset][model]["f1_mean"].append(row["test_macro_f1_mean"])
        organized[dataset][model]["f1_std"].append(row["test_macro_f1_std"])

    # sort by depth
    for dataset in organized:
        for model in organized[dataset]:
            zipped = list(zip(
                organized[dataset][model]["depths"],
                organized[dataset][model]["acc_mean"],
                organized[dataset][model]["acc_std"],
                organized[dataset][model]["f1_mean"],
                organized[dataset][model]["f1_std"],
            ))
            zipped.sort(key=lambda x: x[0])

            depths, acc_mean, acc_std, f1_mean, f1_std = zip(*zipped)

            organized[dataset][model]["depths"] = list(depths)
            organized[dataset][model]["acc_mean"] = list(acc_mean)
            organized[dataset][model]["acc_std"] = list(acc_std)
            organized[dataset][model]["f1_mean"] = list(f1_mean)
            organized[dataset][model]["f1_std"] = list(f1_std)

    return organized


def make_plot(x_gcn, y_gcn, err_gcn, x_sage, y_sage, err_sage, ylabel, title, out_file):
    plt.figure(figsize=(7, 5))

    plt.errorbar(
        x_gcn, y_gcn, yerr=err_gcn,
        marker="o", capsize=4, linewidth=2, label="GCN"
    )
    plt.errorbar(
        x_sage, y_sage, yerr=err_sage,
        marker="s", capsize=4, linewidth=2, label="GraphSAGE"
    )

    plt.xlabel("Depth")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(set(x_gcn + x_sage)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to depth experiment results.json"
    )
    args = parser.parse_args()

    results = load_results(args.results)
    organized = organize_results(results)

    out_dir = os.path.join("experiments", "depth_plots")
    os.makedirs(out_dir, exist_ok=True)

    for dataset in organized:
        if "GCN" not in organized[dataset] or "GraphSAGE" not in organized[dataset]:
            print(f"Skipping {dataset}: missing GCN or GraphSAGE results")
            continue

        gcn = organized[dataset]["GCN"]
        sage = organized[dataset]["GraphSAGE"]

        # Accuracy plot
        make_plot(
            x_gcn=gcn["depths"],
            y_gcn=gcn["acc_mean"],
            err_gcn=gcn["acc_std"],
            x_sage=sage["depths"],
            y_sage=sage["acc_mean"],
            err_sage=sage["acc_std"],
            ylabel="Test Accuracy",
            title=f"{dataset}: Accuracy vs Depth",
            out_file=os.path.join(out_dir, f"{dataset.lower()}_accuracy_vs_depth.png"),
        )

        # Macro-F1 plot
        make_plot(
            x_gcn=gcn["depths"],
            y_gcn=gcn["f1_mean"],
            err_gcn=gcn["f1_std"],
            x_sage=sage["depths"],
            y_sage=sage["f1_mean"],
            err_sage=sage["f1_std"],
            ylabel="Test Macro-F1",
            title=f"{dataset}: Macro-F1 vs Depth",
            out_file=os.path.join(out_dir, f"{dataset.lower()}_macrof1_vs_depth.png"),
        )

    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
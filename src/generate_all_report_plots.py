import os
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# CONFIG: update these paths if your folders differ
# =========================================================
PATHS = {
    "final_comparison": {
        "Cora": {
            "GCN3": "experiments/final_cora_gcn3/final.json",
            "GraphSAGE": "experiments/final_cora_graphsage/final.json",
            "GAT": "experiments/final_cora_gat/final.json",
            "APPNP": "experiments/final_cora_appnp/final.json",
        },
        "PubMed": {
            "GCN3": "experiments/final_pubmed_gcn3/final.json",
            "GraphSAGE": "experiments/final_pubmed_graphsage/final.json",
            "GAT": "experiments/final_pubmed_gat/final.json",
            "APPNP": "experiments/final_pubmed_appnp/final.json",
        },
    },
    "aggregator_results": "experiments/agg_graphsage",  # folder with timestamped runs
    "feature_noise_results": "experiments/feature_noise_robustness",  # folder with timestamped runs
    "edge_dropout_results": "experiments/edge_dropout_robustness",  # folder with timestamped runs
    "depth_results": "experiments/depth_gcn_vs_graphsage",  # folder with timestamped runs
    "node_degree_results": "experiments/node_degree_analysis",  # folder with timestamped runs
    "parameter_efficiency_results": "experiments/parameter_efficiency/results.json",
}

OUT_DIR = "results/plots"


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def find_latest_results_file(parent_dir: str, filename: str = "results.json") -> str:
    p = Path(parent_dir)
    if p.is_file():
        return str(p)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {parent_dir}")

    candidates = [x / filename for x in p.iterdir() if x.is_dir() and (x / filename).exists()]
    if not candidates:
        # also allow parent_dir/results.json directly
        direct = p / filename
        if direct.exists():
            return str(direct)
        raise FileNotFoundError(f"No {filename} found under: {parent_dir}")

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def safe_float(x):
    return None if x is None else float(x)


def save_bar_plot(labels, values, title, ylabel, out_file, errs=None, rotation=0):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    if errs is not None:
        plt.bar(x, values, yerr=errs, capsize=4)
    else:
        plt.bar(x, values)
    plt.xticks(x, labels, rotation=rotation)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


def save_line_plot(series_dict, x_label, y_label, title, out_file):
    plt.figure(figsize=(8, 5))
    for name, series in series_dict.items():
        xs = series["x"]
        ys = series["y"]
        yerr = series.get("yerr")
        if yerr is not None:
            plt.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4, label=name)
        else:
            plt.plot(xs, ys, marker="o", linewidth=2, label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close()


# =========================================================
# PLOT 1 + 2: FINAL MODEL COMPARISON
# =========================================================
def plot_final_model_comparison():
    datasets = ["Cora", "PubMed"]

    for dataset in datasets:
        labels, accs, acc_stds, f1s, f1_stds = [], [], [], [], []

        for model, path in PATHS["final_comparison"][dataset].items():
            if not os.path.exists(path):
                print(f"Missing file: {path}")
                continue
            d = load_json(path)
            labels.append(model)
            accs.append(d["test_acc_mean"])
            acc_stds.append(d["test_acc_std"])
            f1s.append(d["test_macro_f1_mean"])
            f1_stds.append(d["test_macro_f1_std"])

        if labels:
            save_bar_plot(
                labels, accs,
                title=f"{dataset}: Final Accuracy Comparison",
                ylabel="Test Accuracy",
                out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_final_accuracy_comparison.png"),
                errs=acc_stds,
            )

            save_bar_plot(
                labels, f1s,
                title=f"{dataset}: Final Macro-F1 Comparison",
                ylabel="Test Macro-F1",
                out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_final_macrof1_comparison.png"),
                errs=f1_stds,
            )


# =========================================================
# PLOT 3: GRAPH SAGE AGGREGATOR COMPARISON
# =========================================================
def plot_aggregator_comparison():
    results_path = find_latest_results_file(PATHS["aggregator_results"])
    results = load_json(results_path)

    for dataset in ["Cora", "PubMed"]:
        rows = [r for r in results if r["dataset"] == dataset]
        rows.sort(key=lambda x: x["aggr"])

        labels = [r["aggr"] for r in rows]
        accs = [r["test_acc_mean"] for r in rows]
        acc_stds = [r["test_acc_std"] for r in rows]
        f1s = [r["test_macro_f1_mean"] for r in rows]
        f1_stds = [r["test_macro_f1_std"] for r in rows]

        save_bar_plot(
            labels, accs,
            title=f"{dataset}: GraphSAGE Aggregator Comparison (Accuracy)",
            ylabel="Test Accuracy",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_aggregator_accuracy.png"),
            errs=acc_stds,
        )

        save_bar_plot(
            labels, f1s,
            title=f"{dataset}: GraphSAGE Aggregator Comparison (Macro-F1)",
            ylabel="Test Macro-F1",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_aggregator_macrof1.png"),
            errs=f1_stds,
        )


# =========================================================
# PLOT 4: FEATURE NOISE ROBUSTNESS
# =========================================================
def plot_feature_noise():
    results_path = find_latest_results_file(PATHS["feature_noise_results"])
    results = load_json(results_path)

    for dataset in ["Cora", "PubMed"]:
        dataset_rows = [r for r in results if r["dataset"] == dataset]

        series = {}
        for model in ["GCN", "GraphSAGE"]:
            rows = [r for r in dataset_rows if r["model"] == model]
            rows.sort(key=lambda x: x["noise_std"])
            series[model] = {
                "x": [r["noise_std"] for r in rows],
                "y": [r["summary"]["test_acc_mean"] for r in rows],
                "yerr": [r["summary"]["test_acc_std"] for r in rows],
            }

        save_line_plot(
            series,
            x_label="Noise Std",
            y_label="Test Accuracy",
            title=f"{dataset}: Feature Noise Robustness",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_feature_noise_accuracy.png"),
        )

        series_f1 = {}
        for model in ["GCN", "GraphSAGE"]:
            rows = [r for r in dataset_rows if r["model"] == model]
            rows.sort(key=lambda x: x["noise_std"])
            series_f1[model] = {
                "x": [r["noise_std"] for r in rows],
                "y": [r["summary"]["test_macro_f1_mean"] for r in rows],
                "yerr": [r["summary"]["test_macro_f1_std"] for r in rows],
            }

        save_line_plot(
            series_f1,
            x_label="Noise Std",
            y_label="Test Macro-F1",
            title=f"{dataset}: Feature Noise Robustness",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_feature_noise_macrof1.png"),
        )


# =========================================================
# PLOT 5: EDGE DROPOUT ROBUSTNESS
# =========================================================
def plot_edge_dropout():
    results_path = find_latest_results_file(PATHS["edge_dropout_results"])
    results = load_json(results_path)

    for dataset in ["Cora", "PubMed"]:
        dataset_rows = [r for r in results if r["dataset"] == dataset]

        series = {}
        for model in ["GCN", "GraphSAGE"]:
            rows = [r for r in dataset_rows if r["model"] == model]
            rows.sort(key=lambda x: x["edge_drop_p"])
            series[model] = {
                "x": [r["edge_drop_p"] for r in rows],
                "y": [r["summary"]["test_acc_mean"] for r in rows],
                "yerr": [r["summary"]["test_acc_std"] for r in rows],
            }

        save_line_plot(
            series,
            x_label="Edge Drop Probability",
            y_label="Test Accuracy",
            title=f"{dataset}: Edge Dropout Robustness",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_edge_dropout_accuracy.png"),
        )

        series_f1 = {}
        for model in ["GCN", "GraphSAGE"]:
            rows = [r for r in dataset_rows if r["model"] == model]
            rows.sort(key=lambda x: x["edge_drop_p"])
            series_f1[model] = {
                "x": [r["edge_drop_p"] for r in rows],
                "y": [r["summary"]["test_macro_f1_mean"] for r in rows],
                "yerr": [r["summary"]["test_macro_f1_std"] for r in rows],
            }

        save_line_plot(
            series_f1,
            x_label="Edge Drop Probability",
            y_label="Test Macro-F1",
            title=f"{dataset}: Edge Dropout Robustness",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_edge_dropout_macrof1.png"),
        )


# =========================================================
# PLOT 6: DEPTH COMPARISON
# =========================================================
def plot_depth_comparison():
    results_path = find_latest_results_file(PATHS["depth_results"])
    results = load_json(results_path)

    for dataset in ["Cora", "PubMed"]:
        dataset_rows = [r for r in results if r["dataset"] == dataset]

        series_acc = {}
        series_f1 = {}

        for model in ["GCN", "GraphSAGE"]:
            rows = [r for r in dataset_rows if r["model"] == model]
            rows.sort(key=lambda x: x["depth"])
            series_acc[model] = {
                "x": [r["depth"] for r in rows],
                "y": [r["test_acc_mean"] for r in rows],
                "yerr": [r["test_acc_std"] for r in rows],
            }
            series_f1[model] = {
                "x": [r["depth"] for r in rows],
                "y": [r["test_macro_f1_mean"] for r in rows],
                "yerr": [r["test_macro_f1_std"] for r in rows],
            }

        save_line_plot(
            series_acc,
            x_label="Depth",
            y_label="Test Accuracy",
            title=f"{dataset}: Accuracy vs Depth",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_depth_accuracy.png"),
        )

        save_line_plot(
            series_f1,
            x_label="Depth",
            y_label="Test Macro-F1",
            title=f"{dataset}: Macro-F1 vs Depth",
            out_file=os.path.join(OUT_DIR, f"{dataset.lower()}_depth_macrof1.png"),
        )


# =========================================================
# PLOT 7: NODE DEGREE ANALYSIS
# =========================================================
def plot_node_degree_analysis():
    results_path = find_latest_results_file(PATHS["node_degree_results"])
    results = load_json(results_path)

    for dataset in ["Cora", "PubMed"]:
        dataset_rows = [r for r in results if r["dataset"] == dataset]
        labels = ["Low", "Medium", "High"]

        for metric_name, out_suffix, title_suffix in [
            ("acc", "accuracy", "Accuracy by Node Degree"),
            ("f1", "macrof1", "Macro-F1 by Node Degree"),
        ]:
            plt.figure(figsize=(8, 5))
            x = np.arange(len(labels))
            width = 0.35

            for idx, model in enumerate(["GCN", "GraphSAGE"]):
                row = next(r for r in dataset_rows if r["model"] == model)
                if metric_name == "acc":
                    vals = [
                        row["summary"]["low_degree_acc_mean"],
                        row["summary"]["medium_degree_acc_mean"],
                        row["summary"]["high_degree_acc_mean"],
                    ]
                    errs = [
                        row["summary"]["low_degree_acc_std"],
                        row["summary"]["medium_degree_acc_std"],
                        row["summary"]["high_degree_acc_std"],
                    ]
                else:
                    vals = [
                        row["summary"]["low_degree_f1_mean"],
                        row["summary"]["medium_degree_f1_mean"],
                        row["summary"]["high_degree_f1_mean"],
                    ]
                    errs = [
                        row["summary"]["low_degree_f1_std"],
                        row["summary"]["medium_degree_f1_std"],
                        row["summary"]["high_degree_f1_std"],
                    ]

                offset = -width / 2 if idx == 0 else width / 2
                plt.bar(x + offset, vals, width=width, yerr=errs, capsize=4, label=model)

            plt.xticks(x, labels)
            plt.ylabel("Value")
            plt.title(f"{dataset}: {title_suffix}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{dataset.lower()}_node_degree_{out_suffix}.png"), dpi=220)
            plt.close()


# =========================================================
# PLOT 8: PARAMETER EFFICIENCY
# =========================================================
def plot_parameter_efficiency():
    results = load_json(PATHS["parameter_efficiency_results"])

    for dataset in ["Cora", "PubMed"]:
        rows = [r for r in results if r["dataset"] == dataset]

        plt.figure(figsize=(8, 5))
        for row in rows:
            x = row["num_parameters"]
            y = row["test_acc_mean"]
            plt.scatter(x, y, s=90)
            plt.annotate(row["model"], (x, y), xytext=(5, 5), textcoords="offset points")

        plt.xlabel("Number of Parameters")
        plt.ylabel("Test Accuracy")
        plt.title(f"{dataset}: Parameter Efficiency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{dataset.lower()}_parameter_efficiency.png"), dpi=220)
        plt.close()


def main():
    ensure_dir(OUT_DIR)

    plot_final_model_comparison()
    plot_aggregator_comparison()
    plot_feature_noise()
    plot_edge_dropout()
    plot_depth_comparison()
    plot_node_degree_analysis()
    plot_parameter_efficiency()

    print(f"All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
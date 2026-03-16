# Comparative Analysis of GCN and GraphSAGE for Node Classification

This repository contains the implementation and experiments for a case study comparing Graph Neural Network (GNN) architectures for node classification on citation network datasets.

The primary focus of this study is the comparison between **Graph Convolutional Networks (GCN)** and **GraphSAGE**, while additional architectures such as **GAT** and **APPNP** are included as baselines.

---

## Project Overview

Graph Neural Networks are widely used for learning from graph-structured data such as citation networks, social networks, and knowledge graphs.

This project investigates how different GNN architectures behave under various experimental conditions, including:

- baseline performance comparison
- limited labeled data
- inductive-style splits
- model depth analysis
- robustness to noise
- aggregation strategy comparison
- parameter efficiency
- embedding visualization

The experiments provide insights into how architectural design choices influence model performance and representation learning.

---

## Models Implemented

The following Graph Neural Network architectures were implemented using **PyTorch Geometric**:

- Graph Convolutional Networks (GCN)
- GraphSAGE
- Graph Attention Networks (GAT)
- Approximate Personalized Propagation of Neural Predictions (APPNP)

---

## Datasets

Experiments were conducted on the following benchmark datasets:

| Dataset | Nodes | Edges | Features | Classes |
|------|------|------|------|------|
| Cora | 2,708 | 10,556 | 1,433 | 7 |
| PubMed | 19,717 | 88,648 | 500 | 3 |
| OGBN-Arxiv (subset) | 10,000 | 7,580 | 128 | 40 |

The OGBN-Arxiv dataset was used as a **sampled induced subgraph** to enable efficient experimentation.

---

## Experimental Analysis

The study includes the following experiments:

### Baseline Model Comparison
Comparison of GCN, GraphSAGE, GAT, and APPNP.

### Reduced Training Data
Evaluation using only **10% labeled nodes**.

### Inductive-style Split
Analysis of model performance when training data has reduced access to graph structure.

### Hidden Dimension Ablation
Evaluation of different model capacities.

### Node Degree Analysis
Performance comparison across low, medium, and high-degree nodes.

### Oversmoothing Study
Impact of increasing model depth.

### Robustness Experiments
Evaluation with:

- feature noise
- edge dropout

### GraphSAGE Aggregator Comparison
Comparison of:

- mean
- max
- sum aggregators

### Embedding Visualization
t-SNE visualization of learned node embeddings.

---

## Results Summary

Key observations from the experiments:

- **APPNP achieved the strongest baseline performance** on citation datasets.
- **GCN demonstrated the most stable performance across datasets.**
- **GraphSAGE showed advantages in inductive-style scenarios.**
- Increasing model depth leads to **oversmoothing effects**.
- Models remain relatively robust under moderate feature noise and edge dropout.

---

## Project Structure

node-prediction-gcn-graphsage
│
├── src
│   ├── models
│   ├── training scripts
│   └── experiment scripts
│
├── scripts
│
├── notebooks
│
├── data
│
├── results
│
├── dataset_overview.md
├── results_summary.md
│
├── README.md
└── requirements.txt

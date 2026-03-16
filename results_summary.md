# Results Summary

## Architecture Comparison (Best Configurations)

| Dataset | Setting | Model | Test Accuracy | Macro-F1 | Runtime |
|--------|---------|-------|---------------|----------|---------|
| Cora | Transductive | GCN | 0.816 | 0.809 | ~1.2s |
| Cora | Transductive | GraphSAGE | 0.809 | 0.804 | ~2.2s |
| Cora | Transductive | GAT | 0.792 | 0.790 | ~5.6s |
| PubMed | Transductive | GCN | 0.788 | 0.783 | ~2.3s |
| PubMed | Transductive | GraphSAGE | 0.766 | 0.763 | ~11.2s |
| PubMed | Transductive | GAT | 0.767 | 0.763 | ~22s |
| ogbn-arxiv | Inductive | GraphSAGE | 0.133 | – | ~4 min |

**Note:**  
Cora and PubMed use transductive splits.  
OGBN-Arxiv uses inductive splits with neighbor sampling.


## Key Observations

- Hyperparameter tuning improves performance but shows diminishing returns.
- On small and medium transductive citation graphs, GCN is both accurate and computationally efficient.
- GraphSAGE and GAT introduce additional complexity without improving performance in transductive settings.
- GraphSAGE is essential for inductive learning and scalable training, as demonstrated on ogbn-arxiv.

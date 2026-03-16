from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(
    name="ogbn-arxiv",
    root="data"
)

data = dataset[0]

print(data)
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features:", data.num_features)
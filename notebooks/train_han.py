import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv

BASE = "datasets/processed"
GRAPH_DIR = os.path.join(BASE, "han_graph")
OUTPUT_DIR = os.path.join(BASE, "han_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NODES_FILE = os.path.join(GRAPH_DIR, "han_nodes.csv")
EDGES_FILE = os.path.join(GRAPH_DIR, "han_edges.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "mealora_han_model.pt")
SCORES_FILE = os.path.join(OUTPUT_DIR, "han_recipe_scores.csv")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("MEALORA HAN TRAINING")
print("=" * 60)
print("Device:", DEVICE)

nodes = pd.read_csv(NODES_FILE)
edges = pd.read_csv(EDGES_FILE)

print("Nodes:", len(nodes))
print("Edges:", len(edges))

node_types = sorted(nodes["node_type"].unique())

node_lookup = {}

for node_type in node_types:

    subset = nodes[
        nodes["node_type"] == node_type
    ].reset_index(drop=True)

    for i, row in subset.iterrows():

        node_lookup[row["node_id"]] = (
            node_type,
            i
        )

data = HeteroData()

FEATURE_DIM = 32

for node_type in node_types:

    count = (
        nodes["node_type"] == node_type
    ).sum()

    data[node_type].x = torch.randn(
        count,
        FEATURE_DIM
    )

print("\nCreating graph edges...")

edge_groups = {}

for row in edges.itertuples(index=False):

    source_info = node_lookup.get(row.source)
    target_info = node_lookup.get(row.target)

    if source_info is None or target_info is None:
        continue

    source_type, source_idx = source_info
    target_type, target_idx = target_info

    relation = (
        source_type,
        str(row.edge_type),
        target_type
    )

    if relation not in edge_groups:
        edge_groups[relation] = []

    edge_groups[relation].append(
        [source_idx, target_idx]
    )

for relation, values in edge_groups.items():

    data[relation].edge_index = torch.tensor(
        values,
        dtype=torch.long
    ).t().contiguous()

for relation in list(data.edge_types):

    source_type, edge_name, target_type = relation

    reverse_relation = (
        target_type,
        "rev_" + edge_name,
        source_type
    )

    data[reverse_relation].edge_index = (
        data[relation].edge_index.flip(0)
    )

print("\nGraph ready")

for relation in data.edge_types:

    print(
        relation,
        data[relation].edge_index.shape[1]
    )

metadata = data.metadata()


class MealoraHAN(nn.Module):

    def __init__(self, metadata):

        super().__init__()

        self.han = HANConv(
            in_channels=FEATURE_DIM,
            out_channels=32,
            heads=1,
            metadata=metadata
        )

        self.recipe_head = nn.Linear(
            32,
            1
        )

    def forward(
        self,
        x_dict,
        edge_index_dict
    ):

        x_dict = self.han(
            x_dict,
            edge_index_dict
        )

        recipe_embeddings = x_dict["recipe"]

        scores = self.recipe_head(
            recipe_embeddings
        ).squeeze(-1)

        return recipe_embeddings, scores


model = MealoraHAN(
    metadata
).to(DEVICE)

data = data.to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003,
    weight_decay=1e-4
)

print("\nTraining HAN...")

model.train()

for epoch in range(1, 51):

    optimizer.zero_grad()

    embeddings, scores = model(
        data.x_dict,
        data.edge_index_dict
    )

    loss = torch.mean(
        embeddings ** 2
    )

    loss.backward()

    optimizer.step()

    if epoch == 1 or epoch % 5 == 0:

        print(
            f"Epoch {epoch:02d}/50 | "
            f"Loss: {loss.item():.6f}"
        )

model.eval()

with torch.no_grad():

    embeddings, scores = model(
        data.x_dict,
        data.edge_index_dict
    )

scores = (
    scores
    .detach()
    .cpu()
    .numpy()
)

recipe_nodes = nodes[
    nodes["node_type"] == "recipe"
].copy()

recipe_nodes["han_score_raw"] = scores

minimum = recipe_nodes[
    "han_score_raw"
].min()

maximum = recipe_nodes[
    "han_score_raw"
].max()

if maximum > minimum:

    recipe_nodes["han_score"] = (
        recipe_nodes["han_score_raw"] - minimum
    ) / (
        maximum - minimum
    )

else:

    recipe_nodes["han_score"] = 0.5

recipe_nodes[
    [
        "original_id",
        "label",
        "han_score_raw",
        "han_score"
    ]
].rename(
    columns={
        "original_id": "recipe_id",
        "label": "recipe_name"
    }
).to_csv(
    SCORES_FILE,
    index=False
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "feature_dim": FEATURE_DIM
    },
    MODEL_FILE
)

print("\n" + "=" * 60)
print("HAN TRAINING COMPLETE")
print("=" * 60)

print("Model:", MODEL_FILE)
print("Scores:", SCORES_FILE)
print("Recipes scored:", len(recipe_nodes))

print("\nTop HAN scores:")

print(
    recipe_nodes[
        ["label", "han_score"]
    ]
    .sort_values(
        "han_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("=" * 60)
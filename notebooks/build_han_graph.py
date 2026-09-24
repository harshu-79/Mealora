import os
import pandas as pd
import numpy as np
import ast
import json

BASE = "datasets/processed"

RECIPE_FILE = os.path.join(BASE, "recipe_master_dataset.csv")
INTERACTION_FILE = os.path.join(BASE, "user_recipe_interactions.csv")

OUTPUT_DIR = os.path.join(BASE, "han_graph")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MEALORA HAN GRAPH DATASET BUILDER")
print("=" * 60)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

recipes = pd.read_csv(RECIPE_FILE)
interactions = pd.read_csv(INTERACTION_FILE)

print(f"Recipes loaded: {len(recipes)}")
print(f"Interactions loaded: {len(interactions)}")

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def split_values(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    # Try Python-list format
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]
        except Exception:
            pass

    # Try common separators
    for sep in ["|", ";"]:
        if sep in value:
            return [
                x.strip()
                for x in value.split(sep)
                if x.strip()
            ]

    return [value] if value else []


def clean(value):
    return str(value).strip().lower()


# ---------------------------------------------------------
# NODE CREATION
# ---------------------------------------------------------

nodes = []
edges = []

node_ids = {}

def add_node(node_type, original_id, label):
    key = (node_type, str(original_id))

    if key not in node_ids:
        node_id = f"{node_type}_{len(node_ids)}"

        node_ids[key] = node_id

        nodes.append({
            "node_id": node_id,
            "node_type": node_type,
            "original_id": str(original_id),
            "label": str(label)
        })

    return node_ids[key]


def add_edge(source, target, edge_type):
    edges.append({
        "source": source,
        "target": target,
        "edge_type": edge_type
    })


# ---------------------------------------------------------
# RECIPE GRAPH
# ---------------------------------------------------------

print("\nBuilding recipe nodes and relationships...")

for _, row in recipes.iterrows():

    recipe_id = row["recipe_id"]

    recipe_node = add_node(
        "recipe",
        recipe_id,
        row["recipe_name"]
    )

    # -------------------------
    # INGREDIENTS
    # -------------------------

    ingredient_values = split_values(
        row.get("canonical_ingredients", "")
    )

    if not ingredient_values:
        ingredient_values = split_values(
            row.get("ingredient_list", "")
        )

    for ingredient in ingredient_values:

        ingredient_clean = clean(ingredient)

        ingredient_node = add_node(
            "ingredient",
            ingredient_clean,
            ingredient_clean
        )

        add_edge(
            recipe_node,
            ingredient_node,
            "contains"
        )

    # -------------------------
    # CUISINE
    # -------------------------

    if pd.notna(row.get("cuisine")):

        cuisine = clean(row["cuisine"])

        cuisine_node = add_node(
            "cuisine",
            cuisine,
            cuisine
        )

        add_edge(
            recipe_node,
            cuisine_node,
            "belongs_to"
        )

    # -------------------------
    # DIET
    # -------------------------

    if pd.notna(row.get("diet_type")):

        diet = clean(row["diet_type"])

        diet_node = add_node(
            "diet",
            diet,
            diet
        )

        add_edge(
            recipe_node,
            diet_node,
            "suitable_for"
        )

    # -------------------------
    # MEAL TYPE
    # -------------------------

    if pd.notna(row.get("meal_type")):

        meal = clean(row["meal_type"])

        meal_node = add_node(
            "meal_type",
            meal,
            meal
        )

        add_edge(
            recipe_node,
            meal_node,
            "is_type"
        )

    # -------------------------
    # REGION
    # -------------------------

    region_value = row.get("region")

    if pd.notna(region_value):

        region = clean(region_value)

        region_node = add_node(
            "region",
            region,
            region
        )

        add_edge(
            recipe_node,
            region_node,
            "belongs_to_region"
        )

    # -------------------------
    # STATE
    # -------------------------

    state_value = row.get("state")

    if pd.notna(state_value):

        state = clean(state_value)

        state_node = add_node(
            "state",
            state,
            state
        )

        add_edge(
            recipe_node,
            state_node,
            "belongs_to_state"
        )


# ---------------------------------------------------------
# USER → RECIPE INTERACTIONS
# ---------------------------------------------------------

print("Adding user interaction relationships...")

for _, row in interactions.iterrows():

    user_id = row["user_id"]
    recipe_id = row["recipe_id"]

    user_node = add_node(
        "user",
        user_id,
        user_id
    )

    recipe_node = add_node(
        "recipe",
        recipe_id,
        recipe_id
    )

    interaction = clean(row.get("interaction", ""))

    rating = row.get("rating", 0)

    if interaction:
        edge_type = interaction
    else:
        edge_type = "interacted"

    add_edge(
        user_node,
        recipe_node,
        edge_type
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

nodes_df = pd.DataFrame(nodes)
edges_df = pd.DataFrame(edges)

nodes_file = os.path.join(
    OUTPUT_DIR,
    "han_nodes.csv"
)

edges_file = os.path.join(
    OUTPUT_DIR,
    "han_edges.csv"
)

nodes_df.to_csv(nodes_file, index=False)
edges_df.to_csv(edges_file, index=False)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HAN GRAPH CREATED")
print("=" * 60)

print(f"Total nodes : {len(nodes_df)}")
print(f"Total edges : {len(edges_df)}")

print("\nNODE TYPES")
print(nodes_df["node_type"].value_counts())

print("\nEDGE TYPES")
print(edges_df["edge_type"].value_counts())

print("\nFiles created:")

print(nodes_file)
print(edges_file)

print("=" * 60)

import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

DATA = "datasets/processed/recipe_master_dataset.csv"
FEATURES = "datasets/processed/recipe_feature_matrix.npz"
PANTRY = "datasets/processed/recipe_pantry_features.csv"
EXPIRY = "datasets/processed/recipe_expiry_features.csv"


def recommend(
    reference_recipe,
    top_k=2,
):
    df = pd.read_csv(DATA, low_memory=False)
    X = load_npz(FEATURES)

    pantry = pd.read_csv(PANTRY)
    expiry = pd.read_csv(EXPIRY)

    matches = df[
        df["recipe_name"].str.contains(
            reference_recipe,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        raise ValueError("Reference recipe not found.")

    idx = matches.index[0]

    content = cosine_similarity(
        X[idx],
        X
    ).flatten()

    pantry_map = pantry.set_index(
        "recipe_id"
    )["pantry_match_pct"]

    expiry_map = expiry.set_index(
        "recipe_id"
    )["expiry_score"]

    df["content_score"] = content

    df["pantry_score"] = (
        df["recipe_id"]
        .map(pantry_map)
        .fillna(0)
    )

    df["expiry_score"] = (
        df["recipe_id"]
        .map(expiry_map)
        .fillna(0)
    )

    df["final_score"] = (
        0.50 * df["content_score"]
        + 0.30 * df["pantry_score"]
        + 0.20 * df["expiry_score"]
    )

    result = (
        df[
            (df["recipe_scope"] == "indian") &
            (df.index != idx)
        ]
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_k)
    )

    return result[
        [
            "recipe_name",
            "content_score",
            "pantry_score",
            "expiry_score",
            "final_score",
        ]
    ]


if __name__ == "__main__":
    result = recommend(
        "Balu shahi",
        top_k=2
    )

    print(result.to_string(index=False))

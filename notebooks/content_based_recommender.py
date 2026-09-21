import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

DATA = "datasets/processed/recipe_master_dataset.csv"
FEATURES = "datasets/processed/recipe_feature_matrix.npz"

df = pd.read_csv(DATA, low_memory=False)
X = load_npz(FEATURES)


def recommend_similar(recipe_name, top_k=5):

    matches = df[
        df["recipe_name"]
        .str.contains(
            recipe_name,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return None

    index = matches.index[0]

    scores = cosine_similarity(
        X[index],
        X
    ).flatten()

    scores[index] = -1

    top_indices = (
        scores.argsort()[-top_k:][::-1]
    )

    results = df.iloc[top_indices].copy()

    results["similarity_score"] = [
        float(scores[i])
        for i in top_indices
    ]

    return results[
        [
            "recipe_name",
            "cuisine",
            "meal_type",
            "similarity_score"
        ]
    ]


if __name__ == "__main__":

    result = recommend_similar(
        "Balu shahi",
        5
    )

    if result is None:
        print("Recipe not found.")
    else:
        print(result.to_string(index=False))

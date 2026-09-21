import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = "datasets/processed/final_recommendations.csv"
GRAPH_DIR = "datasets/processed/ml_graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)

# =========================================================
# LOAD FINAL RECOMMENDATIONS
# =========================================================

if not os.path.exists(RESULTS):
    print("ERROR: final_recommendations.csv not found.")
    print("Run final_ranker.py first.")
    exit()

df = pd.read_csv(RESULTS)

print("=" * 70)
print("MEALORA FINAL RECOMMENDATION VALIDATION")
print("=" * 70)

print("\nTotal recommendations:", len(df))

# =========================================================
# TOP 10
# =========================================================

print("\n" + "=" * 70)
print("TOP 10 RECOMMENDATIONS")
print("=" * 70)

columns = [
    "recipe_id",
    "recipe_name",
    "content_score",
    "collaborative_score",
    "pantry_score",
    "expiry_score",
    "context_score",
    "hgb_suitability_score",
    "final_score"
]

available = [c for c in columns if c in df.columns]

top10 = df.sort_values(
    "final_score",
    ascending=False
).head(10)

print(
    top10[available].to_string(index=False)
)

# =========================================================
# SCORE SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("AVERAGE COMPONENT SCORES")
print("=" * 70)

score_columns = [
    "content_score",
    "collaborative_score",
    "pantry_score",
    "expiry_score",
    "context_score",
    "hgb_suitability_score",
    "final_score"
]

score_columns = [
    c for c in score_columns
    if c in df.columns
]

print(
    df[score_columns]
    .mean()
    .sort_values(ascending=False)
    .to_string()
)

# =========================================================
# TOP 10 FINAL SCORE GRAPH
# =========================================================

plot_df = top10.copy()

plot_df = plot_df.sort_values(
    "final_score",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["recipe_name"].str[:45],
    plot_df["final_score"]
)

plt.xlabel("Final Recommendation Score")
plt.ylabel("Recipe")

plt.title(
    "Mealora — Top 10 Recommendation Scores",
    fontsize=17,
    fontweight="bold"
)

plt.xlim(0, 1)

plt.tight_layout()

graph_path = (
    f"{GRAPH_DIR}/top_10_recommendation_scores.png"
)

plt.savefig(
    graph_path,
    dpi=200
)

plt.close()

# =========================================================
# SAVE VALIDATION REPORT
# =========================================================

REPORT = "datasets/processed/final_recommendation_validation.txt"

with open(REPORT, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("MEALORA FINAL RECOMMENDATION VALIDATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total recommendations: {len(df)}\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("TOP 10 RECOMMENDATIONS\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        top10[available].to_string(index=False)
    )

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("AVERAGE COMPONENT SCORES\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        df[score_columns]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )

    f.write("\n")

# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(REPORT)
print(graph_path)

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION VALIDATION COMPLETE")
print("=" * 70)

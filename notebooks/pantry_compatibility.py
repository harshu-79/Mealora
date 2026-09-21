import pandas as pd
import ast

DATA = "datasets/processed/recipe_master_dataset.csv"

df = pd.read_csv(DATA, low_memory=False)


def get_ingredients(value):
    try:
        return set(ast.literal_eval(str(value)))
    except:
        return set()


def score_recipe(pantry_items, recipe_ingredients):
    import re
    pantry={str(x).strip().lower() for x in pantry_items}
    ingredients=get_ingredients(recipe_ingredients)
    if not ingredients: return 0.0, 0
    def words(x): return set(re.findall(r"[a-z]+", str(x).lower()))
    available=0
    for ingredient in ingredients:
        iw=words(ingredient)
        if any(pw in iw or any(w in iw for w in words(pw)) for pw in pantry): available += 1
    return available/len(ingredients), len(ingredients)-available



if __name__ == "__main__":

    pantry = [
        "potato",
        "onion",
        "tomato",
        "rice",
    ]

    results = []

    for _, row in df[df["recipe_scope"] == "indian"].iterrows():

        match_pct, missing = score_recipe(
            pantry,
            row["canonical_ingredients"]
        )

        results.append({
            "recipe_name":
                row["recipe_name"],
            "pantry_match_pct":
                round(match_pct, 3),
            "missing_ingredient_count":
                missing,
        })

    result_df = (
        pd.DataFrame(results)
        .sort_values(
            [
                "pantry_match_pct",
                "missing_ingredient_count",
            ],
            ascending=[False, True]
        )
        .head(10)
    )

    print(
        result_df.to_string(index=False)
    )

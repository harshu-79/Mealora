"""
MEALORA WEEKLY MEAL PLANNER

Optimization architecture:
    Existing Mealora RF + HAN recommender
                ↓
       Meal-specific candidates
                ↓
       Feature/utility construction
                ↓
        NSGA-II multi-objective
                ↓
       Pareto-optimal weekly plans
                ↓
     Constraint repair / feasibility
                ↓
            7-day plan

Objectives maximized:
    1. Learned Mealora recommendation suitability (RF + HAN hybrid)
    2. Pantry utilization
    3. Expiry / food-waste reduction
    4. Nutrition quality
    5. Context suitability
    6. Ingredient reuse / shopping reduction
    7. Meal variety

This is deliberately an optimization layer, not another neural network.
The learned RF + HAN recommendation scores remain the ML signals.
"""

import os
import sys
import random
import math
from collections import Counter

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'processed'))
OUT = os.path.join(BASE, 'mealora_weekly_meal_plan.csv')

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MEALS = ['Breakfast', 'Lunch', 'Dinner']
SLOTS = [(d, m) for d in DAYS for m in MEALS]

POPULATION_SIZE = 80
GENERATIONS = 80
ELITE_SIZE = 8
MUTATION_RATE = 0.18
CROSSOVER_RATE = 0.90
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_ranker import recommend


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------

def choose_candidates(meal, diet='Vegetarian', cuisine='South Indian', max_time=60, n=25):
    """Use the existing Mealora RF + HAN recommender as the learned candidate source."""
    result = recommend(
        reference_recipe='Balu shahi',
        target_user=1,
        top_k=n,
        meal_type=meal,
        diet=diet,
        cuisine=cuisine,
        max_time=max_time,
    )
    if result is None:
        return pd.DataFrame()
    return result.copy()


def normalize_ingredients(row):
    columns = [
        'ingredient_names', 'ingredients', 'Cleaned-Ingredients',
        'cleaned_ingredients', 'TranslatedIngredients'
    ]
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col])
            for separator in ['|', ';']:
                value = value.replace(separator, ',')
            return {
                x.strip().lower()
                for x in value.split(',')
                if x.strip() and len(x.strip()) > 1
            }
    return set()


def safe_num(df, col, default=0.5):
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(default).clip(0, 1).to_numpy(dtype=float)


def prepare_candidates(raw):
    if raw.empty:
        return raw

    df = raw.copy()
    if 'recipe_id' not in df.columns:
        df['recipe_id'] = np.arange(len(df))
    if 'recipe_name' not in df.columns:
        df['recipe_name'] = df.get('name', 'Unknown Recipe')

    df = df.drop_duplicates('recipe_id').reset_index(drop=True)

    # Core learned Mealora score.
    final_score = safe_num(df, 'final_score', 0.5)
    rf_score = safe_num(df, 'rf_suitability_score', 0.5)
    han_score = safe_num(df, 'han_score', 0.5)

    # Recommendation components.
    pantry = safe_num(df, 'pantry_score', 0.0)
    expiry = safe_num(df, 'expiry_score', 0.0)
    nutrition = safe_num(df, 'nutrition_score', 0.5)
    context = safe_num(df, 'context_score', 0.5)
    content = safe_num(df, 'content_score', 0.0)

    # Learned recommendation suitability receives the highest weight.
    df['_mealora_utility'] = np.clip(
        0.40 * final_score
        + 0.15 * rf_score
        + 0.10 * han_score
        + 0.10 * content
        + 0.10 * pantry
        + 0.05 * expiry
        + 0.05 * nutrition
        + 0.05 * context,
        0, 1
    )

    df['_pantry'] = pantry
    df['_expiry'] = expiry
    df['_nutrition'] = nutrition
    df['_context'] = context
    df['_rf'] = rf_score
    df['_han'] = han_score
    df['_ingredients'] = [normalize_ingredients(df.iloc[i]) for i in range(len(df))]

    # Meal labels used for slot compatibility.
    meal_col = None
    for col in ['meal_type', 'meal', 'Meal Type', 'mealtype']:
        if col in df.columns:
            meal_col = col
            break
    if meal_col is None:
        df['_meal_text'] = ''
    else:
        df['_meal_text'] = df[meal_col].fillna('').astype(str).str.lower()

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Weekly objective functions
# -----------------------------------------------------------------------------

def plan_features(chromosome, df):
    selected = [df.iloc[i] for i in chromosome]

    utilities = np.array([float(r['_mealora_utility']) for r in selected])
    pantry = np.array([float(r['_pantry']) for r in selected])
    expiry = np.array([float(r['_expiry']) for r in selected])
    nutrition = np.array([float(r['_nutrition']) for r in selected])
    context = np.array([float(r['_context']) for r in selected])

    ingredient_sets = [r['_ingredients'] for r in selected]
    all_ingredients = set().union(*ingredient_sets) if ingredient_sets else set()

    # Ingredient reuse means the weekly plan can reuse ingredients rather than
    # forcing a completely different shopping list every day.
    ingredient_counts = Counter()
    for ingredients in ingredient_sets:
        for ingredient in ingredients:
            ingredient_counts[ingredient] += 1

    if all_ingredients:
        reuse_ratio = np.mean([
            min(ingredient_counts[x] / 3.0, 1.0)
            for x in all_ingredients
        ])
    else:
        reuse_ratio = 0.0

    # Variety: reward different recipes and different ingredient profiles.
    unique_recipes = len(set(chromosome))
    variety = unique_recipes / max(len(chromosome), 1)

    # Meal-to-meal ingredient overlap is useful, but excessive overlap is not.
    pair_overlaps = []
    for i in range(len(ingredient_sets)):
        for j in range(i + 1, len(ingredient_sets)):
            a, b = ingredient_sets[i], ingredient_sets[j]
            if a and b:
                pair_overlaps.append(len(a & b) / max(len(a | b), 1))
    reuse_quality = float(np.mean(pair_overlaps)) if pair_overlaps else reuse_ratio

    # Penalize excessive repetition even if a chromosome is invalid.
    repeat_penalty = 1.0 - (len(chromosome) - unique_recipes) / max(len(chromosome), 1)

    # Balanced nutrition across the week: reward both quality and consistency.
    nutrition_balance = float(np.clip(1.0 - np.std(nutrition), 0, 1))
    nutrition_quality = float(0.75 * np.mean(nutrition) + 0.25 * nutrition_balance)

    # Food-waste objective combines expiry and pantry use.
    waste_reduction = float(0.60 * np.mean(expiry) + 0.40 * np.mean(pantry))

    # Total weekly objectives, all maximized.
    objectives = np.array([
        float(np.mean(utilities)),
        waste_reduction,
        nutrition_quality,
        float(np.mean(context)),
        reuse_quality,
        variety,
        repeat_penalty,
    ])

    return objectives


def constraint_penalty(chromosome, domains):
    penalty = 0.0
    seen = set()

    for slot_idx, recipe_idx in enumerate(chromosome):
        if recipe_idx not in domains[slot_idx]:
            penalty += 2.0
        if recipe_idx in seen:
            penalty += 1.0
        seen.add(recipe_idx)

    return penalty


def fitness(chromosome, df, domains):
    objectives = plan_features(chromosome, df)
    penalty = constraint_penalty(chromosome, domains)

    # NSGA-II uses maximization. Constraint violation is represented by a
    # sufficiently large reduction in every objective.
    if penalty > 0:
        objectives = objectives - penalty

    return objectives


# -----------------------------------------------------------------------------
# NSGA-II implementation
# -----------------------------------------------------------------------------

def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def fast_non_dominated_sort(population, fitnesses):
    domination_sets = [[] for _ in population]
    domination_count = [0 for _ in population]
    fronts = [[]]

    for p in range(len(population)):
        for q in range(len(population)):
            if p == q:
                continue
            if dominates(fitnesses[p], fitnesses[q]):
                domination_sets[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(front, fitnesses):
    if not front:
        return {}
    distance = {i: 0.0 for i in front}
    n_obj = fitnesses.shape[1]

    for obj in range(n_obj):
        ordered = sorted(front, key=lambda i: fitnesses[i, obj])
        distance[ordered[0]] = float('inf')
        distance[ordered[-1]] = float('inf')

        low = fitnesses[ordered[0], obj]
        high = fitnesses[ordered[-1], obj]
        if math.isclose(high, low):
            continue

        for j in range(1, len(ordered) - 1):
            distance[ordered[j]] += (
                fitnesses[ordered[j + 1], obj]
                - fitnesses[ordered[j - 1], obj]
            ) / (high - low)

    return distance


def tournament(population, fitnesses, ranks, crowding):
    a, b = random.sample(range(len(population)), 2)
    if ranks[a] < ranks[b]:
        return population[a].copy()
    if ranks[b] < ranks[a]:
        return population[b].copy()
    return population[a].copy() if crowding[a] >= crowding[b] else population[b].copy()


def repair(chromosome, domains, n_recipes):
    """Repair invalid meal assignments and duplicate recipes."""
    result = chromosome.copy()
    used = set()

    for slot in range(len(result)):
        current = result[slot]
        valid = current in domains[slot] and current not in used
        if valid:
            used.add(current)
            continue

        choices = [x for x in domains[slot] if x not in used]
        if choices:
            result[slot] = random.choice(choices)
            used.add(result[slot])
        else:
            # Last resort: choose any valid domain recipe. The fitness penalty
            # keeps this situation visible to NSGA-II.
            result[slot] = random.choice(domains[slot])
            used.add(result[slot])

    return result


def crossover(parent_a, parent_b, domains, n_recipes):
    if random.random() > CROSSOVER_RATE:
        return parent_a.copy(), parent_b.copy()

    point = random.randint(1, len(parent_a) - 1)
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])

    child_a = repair(child_a, domains, n_recipes)
    child_b = repair(child_b, domains, n_recipes)
    return child_a, child_b


def mutate(chromosome, domains, n_recipes):
    child = chromosome.copy()
    for slot in range(len(child)):
        if random.random() < MUTATION_RATE:
            choices = [x for x in domains[slot] if x != child[slot]]
            if choices:
                child[slot] = random.choice(choices)
    return repair(child, domains, n_recipes)


def nsga2(df, domains):
    n_slots = len(domains)
    n_recipes = len(df)

    if any(len(d) == 0 for d in domains):
        raise RuntimeError('At least one weekly meal slot has no valid candidate recipes.')

    # Ensure the candidate set can satisfy the no-repeat constraint.
    union_size = len(set().union(*domains))
    if union_size < n_slots:
        raise RuntimeError(
            f'Only {union_size} unique recipes are available for {n_slots} meal slots. '
            'Increase candidate pool size.'
        )

    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = np.array([
            random.choice(domain) for domain in domains
        ], dtype=int)
        chromosome = repair(chromosome, domains, n_recipes)
        population.append(chromosome)

    best_front = None
    best_front_fitness = None

    for generation in range(GENERATIONS):
        fitnesses = np.array([
            fitness(ch, df, domains) for ch in population
        ])

        fronts = fast_non_dominated_sort(population, fitnesses)
        ranks = np.zeros(len(population), dtype=int)
        crowd = np.zeros(len(population), dtype=float)

        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            cd = crowding_distance(front, fitnesses)
            for idx, value in cd.items():
                crowd[idx] = value

        if fronts and fronts[0]:
            best_front = [population[i].copy() for i in fronts[0]]
            best_front_fitness = fitnesses[fronts[0]]

        offspring = []
        while len(offspring) < POPULATION_SIZE:
            p1 = tournament(population, fitnesses, ranks, crowd)
            p2 = tournament(population, fitnesses, ranks, crowd)
            c1, c2 = crossover(p1, p2, domains, n_recipes)
            c1 = mutate(c1, domains, n_recipes)
            c2 = mutate(c2, domains, n_recipes)
            offspring.extend([c1, c2])
        offspring = offspring[:POPULATION_SIZE]

        combined = population + offspring
        combined_fitnesses = np.array([
            fitness(ch, df, domains) for ch in combined
        ])

        combined_fronts = fast_non_dominated_sort(combined, combined_fitnesses)
        new_population = []

        for front in combined_fronts:
            if len(new_population) + len(front) <= POPULATION_SIZE:
                new_population.extend([combined[i] for i in front])
            else:
                cd = crowding_distance(front, combined_fitnesses)
                ordered = sorted(front, key=lambda i: cd[i], reverse=True)
                remaining = POPULATION_SIZE - len(new_population)
                new_population.extend([combined[i] for i in ordered[:remaining]])
                break

        population = new_population

        if generation == 0 or (generation + 1) % 10 == 0:
            current = np.array([
                fitness(ch, df, domains) for ch in population
            ])
            print(
                f'Generation {generation + 1:03d}/{GENERATIONS} | '
                f'population={len(population)} | '
                f'best learned suitability={current[:, 0].max():.4f}'
            )

    if not best_front:
        raise RuntimeError('NSGA-II did not produce a feasible Pareto front.')

    # Select the balanced Pareto solution rather than simply maximizing one
    # objective. This is the central multi-objective planning step.
    scores = []
    for i, obj in enumerate(best_front_fitness):
        normalized = np.clip(obj, -1, 1)
        # Weighted decision from the Pareto front.
        balanced = (
            0.30 * normalized[0] +  # Mealora learned suitability
            0.15 * normalized[1] +  # waste reduction
            0.15 * normalized[2] +  # nutrition
            0.10 * normalized[3] +  # context
            0.15 * normalized[4] +  # ingredient reuse
            0.10 * normalized[5] +  # variety
            0.05 * normalized[6]    # uniqueness
        )
        scores.append(balanced)

    selected = best_front[int(np.argmax(scores))]
    return selected, best_front, best_front_fitness


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def create_plan_dataframe(chromosome, df, household_servings=4):
    rows = []
    for slot_idx, (day, meal) in enumerate(SLOTS):
        row = df.iloc[int(chromosome[slot_idx])]
        rows.append({
            'day': day,
            'meal': meal,
            'recipe_id': row.get('recipe_id'),
            'recipe_name': row.get('recipe_name'),
            'mealora_score': round(float(row.get('final_score', 0.0)), 4),
            'rf_suitability_score': round(float(row.get('rf_suitability_score', 0.0)), 4),
            'han_score': round(float(row.get('han_score', 0.0)), 4),
            'pantry_score': round(float(row.get('pantry_score', 0.0)), 4),
            'expiry_score': round(float(row.get('expiry_score', 0.0)), 4),
            'nutrition_score': round(float(row.get('nutrition_score', 0.5)), 4),
            'nutrition_available': int(row.get('nutrition_available', 0)),
            'context_score': round(float(row.get('context_score', 0.5)), 4),
            'household_servings': household_servings,
            'planning_algorithm': 'NSGA-II Multi-Objective Evolutionary Optimization',
        })
    return pd.DataFrame(rows)


def generate_weekly_plan(diet='Vegetarian', cuisine='South Indian', max_time=60, household_servings=4):
    """Generate one optimized 7-day household plan using the existing RF + HAN + NSGA-II pipeline."""
    print('\n' + '=' * 90)
    print('MEALORA WEEKLY MEAL PLANNER — ADVANCED OPTIMIZATION')
    print('=' * 90)
    print('ML signals       : Random Forest + HAN')
    print('Optimizer        : NSGA-II (Multi-Objective Evolutionary Optimization)')
    print('Planning horizon : 7 days × 3 meals = 21 decisions')
    print('Objectives       : suitability, pantry, expiry, nutrition, context, reuse, variety')
    print('Diet             :', diet)
    print('Cuisine          :', cuisine)
    print('Max cooking time :', max_time)
    print('Household servings:', household_servings)
    print('Random seed      :', RANDOM_SEED)

    pools = []
    for meal in MEALS:
        print(f'\nGenerating {meal} candidate pool using Mealora RF + HAN...')
        pool = choose_candidates(
            meal,
            diet=diet,
            cuisine=cuisine,
            max_time=max_time,
            n=25,
        )
        if not pool.empty:
            pool['_requested_meal'] = meal
            pools.append(pool)
            print(f'  {len(pool)} candidates received.')
        else:
            print(f'  WARNING: no candidates returned for {meal}.')

    if not pools:
        raise RuntimeError('No candidate recipes were returned by the Mealora recommender.')

    candidates = pd.concat(pools, ignore_index=True)
    candidates = prepare_candidates(candidates)

    print(f'\nUnique candidate recipes: {len(candidates)}')

    domains = []
    for day, meal in SLOTS:
        requested = candidates['_requested_meal'].astype(str).str.lower()
        mask_requested = requested.eq(meal.lower())

        domain = list(candidates.index[mask_requested])
        if not domain:
            meal_text = candidates['_meal_text'].astype(str)
            domain = list(candidates.index[meal_text.str.contains(meal.lower(), regex=False, na=False)])
        if not domain:
            domain = list(candidates.index)

        domains.append(domain)
        print(f'{day:10s} {meal:9s}: {len(domain)} candidate choices')

    print('\n' + '=' * 90)
    print('RUNNING NSGA-II')
    print('=' * 90)

    chromosome, pareto_front, pareto_fitness = nsga2(candidates, domains)
    plan = create_plan_dataframe(
        chromosome,
        candidates,
        household_servings=household_servings,
    )
    plan.to_csv(OUT, index=False)

    final_objectives = plan_features(chromosome, candidates)

    print('\n' + '=' * 90)
    print('SELECTED PARETO-OPTIMAL WEEKLY PLAN')
    print('=' * 90)
    print(plan[[
        'day', 'meal', 'recipe_name', 'mealora_score',
        'rf_suitability_score', 'han_score', 'pantry_score',
        'expiry_score', 'nutrition_score'
    ]].to_string(index=False))

    print('\n' + '=' * 90)
    print('MULTI-OBJECTIVE RESULTS')
    print('=' * 90)
    names = [
        'Learned suitability', 'Waste reduction', 'Nutrition balance',
        'Context suitability', 'Ingredient reuse', 'Recipe variety', 'Uniqueness'
    ]
    for name, value in zip(names, final_objectives):
        print(f'{name:24s}: {value:.4f}')

    print(f'\nPareto solutions found: {len(pareto_front)}')
    print('Planning algorithm    : NSGA-II')
    print('ML recommendation     : Random Forest + HAN')
    print('Output                :', OUT)
    print('\nDONE.')

    return {
        'plan': plan.to_dict(orient='records'),
        'objectives': {
            name: round(float(value), 4)
            for name, value in zip(names, final_objectives)
        },
        'pareto_solutions': len(pareto_front),
        'planning_algorithm': 'NSGA-II Multi-Objective Evolutionary Optimization',
        'ml_recommendation': 'Random Forest + HAN',
        'household_servings': household_servings,
        'diet': diet,
        'cuisine': cuisine,
        'max_time': max_time,
        'output': OUT,
    }


if __name__ == '__main__':
    generate_weekly_plan()

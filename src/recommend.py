"""
Dish Recommendation Engine for the Food Waste Prediction System.

Hybrid system:
  A) Rule-based filtering (inventory match, menu type, allergens)
  B) XGBoost-based ML ranking (inventory usage, waste reduction, demand alignment)
  C) SHAP explainability for ranked dishes
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.dish_dataset import (
    load_or_generate_dishes,
    load_or_generate_nutrition,
    compute_dish_features,
    build_ingredient_matrix,
)
from src.inventory import (
    parse_inventory,
    compute_batch_match_scores,
    compute_expiry_risk,
    compute_dish_waste_reduction_score,
    estimate_servings,
    parse_quantities,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering for the Ranker
# ─────────────────────────────────────────────────────────────────────────────

RANKER_FEATURE_COLS = [
    'usage_score',
    'waste_reduction_score',
    'demand_alignment',
    'coverage_ratio',
    'quantity_ratio',
    'complexity_score',
    'cost_score',
    'caloric_density',
    'ingredient_count',
]


def build_ranking_features(
    enriched_df: pd.DataFrame,
    inventory: dict,
    predicted_meals: int,
    nutrition_df: pd.DataFrame,
    days_until_expiry: dict = None
) -> pd.DataFrame:
    """
    Enrich the candidate dish DataFrame with all features required by the ranker.

    Returns:
        DataFrame with RANKER_FEATURE_COLS columns (plus dish_name, menu_type, etc.)
    """
    df = enriched_df.copy()

    # Expiry risk per ingredient
    expiry_risks = compute_expiry_risk(inventory, nutrition_df, days_until_expiry)

    waste_scores = []
    demand_aligns = []
    max_servings_list = []

    for _, row in df.iterrows():
        # Waste reduction: how many near-expiry items does this dish use?
        ws = compute_dish_waste_reduction_score(
            row['ingredient_list'], inventory, expiry_risks
        )
        waste_scores.append(ws)

        # Estimated servings & demand alignment
        serv_info = estimate_servings(row, inventory, predicted_meals)
        demand_aligns.append(serv_info['demand_alignment'])
        max_servings_list.append(serv_info['max_servings'])

    df['waste_reduction_score'] = waste_scores
    df['demand_alignment'] = demand_aligns
    df['estimated_servings'] = max_servings_list

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Ranking Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_ranking_labels(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Auto-generate ranking scores from a weighted combination of feature columns.
    Used when no historical ranking labels are available.

    Higher → dish should be ranked higher.
    """
    weights = {
        'usage_score':           0.35,
        'waste_reduction_score': 0.25,
        'demand_alignment':      0.20,
        'coverage_ratio':        0.10,
        '-cost_score':           0.05,   # negative = cheaper is better
        '-complexity_score':     0.05,   # negative = simpler is easier to make
    }

    score = np.zeros(len(feature_df))
    for col_expr, weight in weights.items():
        sign = -1 if col_expr.startswith('-') else 1
        col = col_expr.lstrip('-')
        if col in feature_df.columns:
            col_vals = feature_df[col].fillna(0).values
            score += sign * weight * col_vals

    # Normalise to 0–1
    s_min, s_max = score.min(), score.max()
    if s_max > s_min:
        score = (score - s_min) / (s_max - s_min)

    return score.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ranker(
    dishes_df: pd.DataFrame,
    inventory: dict,
    predicted_meals: int,
    nutrition_df: pd.DataFrame,
    days_until_expiry: dict = None
) -> dict:
    """
    Train an XGBoost ranker on the dish feature set.

    Returns:
        dict: model artifacts for the recommender
    """
    print("\n  🔄 Building ranking feature matrix...")
    enriched = build_ranking_features(
        dishes_df, inventory, predicted_meals, nutrition_df, days_until_expiry
    )

    # Drop rows with any NaN feature
    clean_df = enriched.dropna(subset=RANKER_FEATURE_COLS)
    X = clean_df[RANKER_FEATURE_COLS].values.astype(float)

    print("  🔄 Generating synthetic ranking labels...")
    y = generate_synthetic_ranking_labels(clean_df)

    print(f"  📊 Training on {X.shape[0]} dishes × {X.shape[1]} features")

    model = xgb.XGBRegressor(**config.XGBOOST_RANKER_PARAMS)
    model.fit(X, y)

    artifacts = {
        'model': model,
        'feature_cols': RANKER_FEATURE_COLS,
        'candidate_dishes': clean_df.to_dict('records'),
        'dishes_df': clean_df,
    }

    print("  ✅ XGBoost ranker trained successfully.")
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_candidates(
    dishes_df: pd.DataFrame,
    menu_type: str = 'any',
    min_match: float = None,
    allergens: list = None
) -> pd.DataFrame:
    """
    Apply hard rule-based filters to the candidate dish list.

    Args:
        dishes_df  : enriched DataFrame with usage_score, menu_type columns
        menu_type  : 'veg' | 'non-veg' | 'vegan' | 'any'
        min_match  : minimum ingredient coverage fraction (default config.MIN_INGREDIENT_MATCH)
        allergens  : list of allergen group names to exclude
                     (see config.ALLERGEN_GROUPS keys)

    Returns:
        Filtered DataFrame
    """
    if min_match is None:
        min_match = config.MIN_INGREDIENT_MATCH

    df = dishes_df.copy()

    # 1. Menu type filter
    mt = menu_type.lower().strip() if menu_type else 'any'
    if mt != 'any':
        if mt == 'veg':
            # veg includes veg + vegan
            df = df[df['menu_type'].isin(['veg', 'vegan'])]
        else:
            df = df[df['menu_type'] == mt]

    # 2. Minimum ingredient match
    df = df[df['usage_score'] >= min_match]

    # 3. Allergen exclusion
    if allergens:
        blocked_ings = set()
        for allergen in allergens:
            allergen_lower = allergen.lower()
            ings = config.ALLERGEN_GROUPS.get(allergen_lower, [])
            blocked_ings.update(ings)

        if blocked_ings:
            def has_allergen(row):
                return any(
                    bi in ' '.join(row['ingredient_list'])
                    for bi in blocked_ings
                )
            df = df[~df.apply(has_allergen, axis=1)]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Recommender: Load / Save
# ─────────────────────────────────────────────────────────────────────────────

def load_recommender(filepath: str = None) -> dict:
    """
    Load a previously saved recommender model.

    Returns:
        dict: recommender artifacts (model, feature_cols, candidate_dishes)
    """
    if filepath is None:
        filepath = config.RECOMMENDER_MODEL_FILE

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Recommender model not found at '{filepath}'. "
            "Run `python src/train_recommender.py` first."
        )

    print(f"  📂 Loading recommender from: {filepath}")
    artifacts = joblib.load(filepath)
    print("  ✅ Recommender loaded successfully.")
    return artifacts


def save_recommender(artifacts: dict, filepath: str = None):
    """Save recommender artifacts to disk."""
    if filepath is None:
        filepath = config.RECOMMENDER_MODEL_FILE

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(artifacts, filepath)
    print(f"  💾 Recommender model saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Core Public API
# ─────────────────────────────────────────────────────────────────────────────

def recommend_dishes(
    inventory: dict,
    predicted_meals: int,
    menu_type: str = 'any',
    top_n: int = None,
    allergens: list = None,
    days_until_expiry: dict = None,
    min_match: float = None,
    recommender_path: str = None
) -> list:
    """
    Recommend the top-N dishes to cook given the available inventory.

    Args:
        inventory         : {ingredient: quantity_g}
        predicted_meals   : expected number of meals to serve today
        menu_type         : 'veg' | 'non-veg' | 'vegan' | 'any'
        top_n             : number of recommendations (default config.RECOMMENDER_TOP_N)
        allergens         : list of allergen group names to exclude
        days_until_expiry : {ingredient: days_remaining} (optional)
        min_match         : minimum ingredient coverage (overrides config default)
        recommender_path  : path to saved recommender model

    Returns:
        list of dicts:
            dish_name, inventory_usage_score, waste_reduction_score,
            estimated_servings, confidence_score, missing_ingredients, menu_type
    """
    if top_n is None:
        top_n = config.RECOMMENDER_TOP_N

    # ── Step 1: Load datasets ─────────────────────────────────────────────
    dishes_raw = load_or_generate_dishes()
    nutrition_df = load_or_generate_nutrition()

    # ── Step 2: Compute dish features ────────────────────────────────────
    dishes_feat = compute_dish_features(dishes_raw)

    # ── Step 3: Parse and validate inventory ──────────────────────────────
    inv = parse_inventory(inventory)

    # ── Step 4: Compute inventory match scores ───────────────────────────
    dishes_scored = compute_batch_match_scores(dishes_feat, inv)

    # ── Step 5: Build full ranking features ──────────────────────────────
    dishes_enriched = build_ranking_features(
        dishes_scored, inv, predicted_meals, nutrition_df, days_until_expiry
    )

    # ── Step 6: Rule-based filtering ─────────────────────────────────────
    candidates = filter_candidates(
        dishes_enriched,
        menu_type=menu_type,
        min_match=min_match,
        allergens=allergens
    )

    if candidates.empty:
        print("  ⚠️  No dishes passed the filter. Relaxing min_match threshold...")
        candidates = filter_candidates(
            dishes_enriched,
            menu_type=menu_type,
            min_match=0.10,
            allergens=allergens
        )

    if candidates.empty:
        return []

    # ── Step 7: ML Ranking ────────────────────────────────────────────────
    try:
        artifacts = load_recommender(recommender_path)
        model = artifacts['model']
        feature_cols = artifacts['feature_cols']

        # Use only columns available in candidates
        avail_cols = [c for c in feature_cols if c in candidates.columns]
        X = candidates[avail_cols].fillna(0).values.astype(float)
        scores = model.predict(X)
        candidates = candidates.copy()
        candidates['confidence_score'] = np.clip(scores, 0, 1).round(4)

    except FileNotFoundError:
        # Fallback: rank by usage_score + waste reduction heuristic
        candidates = candidates.copy()
        fallback = (
            0.50 * candidates['usage_score'] +
            0.30 * candidates.get('waste_reduction_score', 0) +
            0.20 * candidates.get('demand_alignment', 0)
        )
        candidates['confidence_score'] = np.clip(fallback, 0, 1).round(4)

    # ── Step 8: Sort & Return Top-N ──────────────────────────────────────
    ranked = candidates.sort_values('confidence_score', ascending=False).head(top_n)

    results = []
    for _, row in ranked.iterrows():
        results.append({
            'dish_name': row['dish_name'],
            'inventory_usage_score': round(float(row['usage_score']), 4),
            'waste_reduction_score': round(float(row.get('waste_reduction_score', 0)), 4),
            'estimated_servings': int(row.get('estimated_servings', row.get('servings', 4))),
            'confidence_score': round(float(row['confidence_score']), 4),
            'menu_type': row['menu_type'],
            'cuisine': row.get('cuisine', 'unknown'),
            'calories_per_serving': int(row.get('calories', 0)),
            'missing_ingredients': row.get('missing_ingredients', []),
            'prep_time_min': int(row.get('prep_time_min', 0)),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────

def explain_dish_ranking(dish_name: str, artifacts: dict, top_n: int = None):
    """
    Explain why a specific dish was ranked the way it was using SHAP.

    Args:
        dish_name : name of the dish to explain
        artifacts : recommender artifacts dict (from load_recommender)
        top_n     : number of features to show (default config.RECOMMENDER_SHAP_TOP_N)

    Returns:
        dict: explanation with SHAP values per feature
    """
    if top_n is None:
        top_n = config.RECOMMENDER_SHAP_TOP_N

    model = artifacts['model']
    feature_cols = artifacts['feature_cols']
    dishes_df = pd.DataFrame(artifacts['candidate_dishes'])

    # Find the target dish
    match = dishes_df[dishes_df['dish_name'].str.lower() == dish_name.lower()]
    if match.empty:
        # Partial match
        match = dishes_df[dishes_df['dish_name'].str.lower().str.contains(
            dish_name.lower(), na=False
        )]

    if match.empty:
        print(f"  ⚠️  Dish '{dish_name}' not found in candidate set.")
        return {}

    row = match.iloc[[0]]
    avail_cols = [c for c in feature_cols if c in row.columns]
    X = row[avail_cols].fillna(0).values.astype(float)

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    base_value = float(explainer.expected_value)
    predicted_score = float(model.predict(X)[0])

    contributions = sorted(
        zip(avail_cols, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\n{'='*60}")
    print(f"🍽️  SHAP Explanation: {match.iloc[0]['dish_name']}")
    print(f"{'='*60}")
    print(f"  📍 Base score   : {base_value:.4f}")
    print(f"  🎯 Final score  : {predicted_score:.4f}")
    print(f"\n  Top {top_n} feature contributions:")
    print(f"  {'Feature':<28} {'SHAP Value':>12}  Impact")
    print(f"  {'-'*50}")

    explanation = {'dish_name': match.iloc[0]['dish_name'], 'features': []}
    for feat, sv in contributions[:top_n]:
        direction = "⬆️ +" if sv > 0 else "⬇️ "
        print(f"  {feat:<28} {sv:>+12.4f}  {direction}")
        explanation['features'].append({'feature': feat, 'shap_value': round(sv, 6)})

    explanation['base_score'] = round(base_value, 4)
    explanation['final_score'] = round(predicted_score, 4)

    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Quick sanity check / demo of the recommender."""
    print("\n" + "="*60)
    print("🍽️  DISH RECOMMENDATION DEMO")
    print("="*60)

    inventory = {
        "rice": 50,
        "potato": 20,
        "onion": 10,
        "tomato": 15,
        "oil": 5,
        "paneer": 8,
        "lentils": 30,
        "turmeric": 2,
        "cumin": 2,
        "garlic": 5,
        "ginger": 5,
        "peas": 10,
    }

    print("\n📦 Inventory:")
    for k, v in inventory.items():
        print(f"   {k}: {v}g")

    results = recommend_dishes(
        inventory=inventory,
        predicted_meals=200,
        menu_type='veg',
        top_n=5
    )

    print(f"\n🏆 Top {len(results)} Recommended Dishes:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['dish_name']} ({r['menu_type']}, {r['cuisine']})")
        print(f"   🔵 Usage Score     : {r['inventory_usage_score']:.2%}")
        print(f"   ♻️  Waste Reduction : {r['waste_reduction_score']:.2%}")
        print(f"   🍽️  Est. Servings   : {r['estimated_servings']}")
        print(f"   🎯 Confidence      : {r['confidence_score']:.4f}")
        if r['missing_ingredients']:
            print(f"   ❌ Missing         : {', '.join(r['missing_ingredients'][:3])}")


if __name__ == '__main__':
    main()

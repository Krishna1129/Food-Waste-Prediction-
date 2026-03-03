"""
Inventory Management Module for the Food Waste Prediction System.

Handles parsing, validation, and scoring of kitchen inventory inputs
relative to dish requirements.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.dish_dataset import parse_ingredients, parse_quantities, load_or_generate_nutrition


# ─────────────────────────────────────────────────────────────────────────────
# Inventory Parsing & Validation
# ─────────────────────────────────────────────────────────────────────────────

def parse_inventory(inventory_dict: dict) -> dict:
    """
    Validate and normalise an inventory dictionary.

    Args:
        inventory_dict: {ingredient_name: quantity_in_g, ...}

    Returns:
        Cleaned dict with lower-cased ingredient names and non-negative quantities.
    """
    if not isinstance(inventory_dict, dict):
        raise ValueError("Inventory must be a dict of {ingredient: quantity_g}.")

    parsed = {}
    for ing, qty in inventory_dict.items():
        ing_clean = str(ing).strip().lower().replace(' ', '_')
        qty_val = max(0.0, float(qty))
        if qty_val > 0:
            parsed[ing_clean] = qty_val

    if not parsed:
        raise ValueError("Inventory is empty or all quantities are zero.")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Ingredient Match Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_inventory_match(
    dish_ingredients: list,
    inventory: dict,
    dish_quantities: Optional[list] = None
) -> dict:
    """
    Compute how well the current inventory covers a dish's ingredient list.

    Args:
        dish_ingredients : list of ingredient names required by the dish
        inventory        : {ingredient: quantity_g} currently available
        dish_quantities  : list of required quantities in grams (same order as
                           dish_ingredients). If None, only binary coverage is used.

    Returns:
        dict with keys:
            - coverage_ratio   : fraction of distinct ingredients present (0–1)
            - quantity_ratio   : how well quantities are satisfied (0–1)
            - missing          : list of missing ingredients
            - available        : list of available ingredients
            - usage_score      : combined score (0–1) used for ranking
    """
    dish_ings = [i.lower().replace(' ', '_') for i in dish_ingredients]
    inventory_keys = set(inventory.keys())

    available = [i for i in dish_ings if i in inventory_keys]
    missing = [i for i in dish_ings if i not in inventory_keys]

    n_total = len(dish_ings)
    coverage_ratio = len(available) / max(n_total, 1)

    # Quantity ratio: for present ingredients, check if we have enough
    if dish_quantities and len(dish_quantities) == n_total:
        qty_scores = []
        for ing, req_qty in zip(dish_ings, dish_quantities):
            avail_qty = inventory.get(ing, 0)
            if req_qty > 0:
                qty_scores.append(min(avail_qty / req_qty, 1.0))
        quantity_ratio = float(np.mean(qty_scores)) if qty_scores else 0.0
    else:
        quantity_ratio = coverage_ratio  # fall back to binary coverage

    # Blend: 70 % coverage, 30 % quantity adequacy
    usage_score = round(0.7 * coverage_ratio + 0.3 * quantity_ratio, 4)

    return {
        'coverage_ratio': round(coverage_ratio, 4),
        'quantity_ratio': round(quantity_ratio, 4),
        'missing': missing,
        'available': available,
        'usage_score': usage_score
    }


def compute_batch_match_scores(
    dishes_df: pd.DataFrame,
    inventory: dict
) -> pd.DataFrame:
    """
    Compute ingredient match scores for all dishes at once.

    Args:
        dishes_df : DataFrame produced by compute_dish_features()
        inventory : parsed inventory dict

    Returns:
        dishes_df with extra columns:
            coverage_ratio, quantity_ratio, usage_score, missing_ingredients
    """
    records = []
    for _, row in dishes_df.iterrows():
        dish_ings = row['ingredient_list']
        dish_qtys = parse_quantities(row['quantities_g'])
        match = score_inventory_match(dish_ings, inventory, dish_qtys)
        records.append({
            'dish_name': row['dish_name'],
            'coverage_ratio': match['coverage_ratio'],
            'quantity_ratio': match['quantity_ratio'],
            'usage_score': match['usage_score'],
            'missing_ingredients': match['missing']
        })

    match_df = pd.DataFrame(records)
    result = dishes_df.merge(match_df, on='dish_name', how='left')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Expiry Risk Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_expiry_risk(
    inventory: dict,
    nutrition_df: Optional[pd.DataFrame] = None,
    days_until_expiry: Optional[dict] = None
) -> dict:
    """
    Assign an urgency/expiry-risk score to each ingredient in inventory.

    Args:
        inventory         : parsed inventory dict
        nutrition_df      : DataFrame with 'ingredient' and 'shelf_life_days'
        days_until_expiry : Optional override {ingredient: days_remaining}

    Returns:
        dict {ingredient: expiry_risk_score (0–1)}, higher → more urgent
    """
    if nutrition_df is None:
        nutrition_df = load_or_generate_nutrition()

    shelf_map = dict(zip(nutrition_df['ingredient'], nutrition_df['shelf_life_days']))

    risk_scores = {}
    for ing in inventory:
        if days_until_expiry and ing in days_until_expiry:
            days_left = days_until_expiry[ing]
        else:
            # Use shelf life as proxy: shorter-lived items = higher urgency
            shelf_life = shelf_map.get(ing, 30)
            days_left = shelf_life  # assume item was just stocked

        # Normalise risk: 0-day expiry → risk 1.0; 365+ days → risk ~0
        risk = max(0.0, 1.0 - (days_left / 365.0))
        risk_scores[ing] = round(risk, 4)

    return risk_scores


def compute_dish_waste_reduction_score(
    dish_ingredients: list,
    inventory: dict,
    expiry_risks: dict
) -> float:
    """
    Score how well a dish uses high-expiry-risk ingredients.
    Higher → dish uses more at-risk items, reducing waste.

    Returns:
        float (0–1)
    """
    dish_ings_clean = [i.lower().replace(' ', '_') for i in dish_ingredients]
    risks = [expiry_risks.get(i, 0.0) for i in dish_ings_clean if i in inventory]

    if not risks:
        return 0.0

    # Weight by presence; average risk of covered ingredients
    return round(float(np.mean(risks)), 4)


def estimate_servings(
    dish_row: pd.Series,
    inventory: dict,
    predicted_meals: int
) -> dict:
    """
    Estimate how many servings can be made given available inventory.

    Returns:
        dict with 'max_servings' and 'demand_alignment' (0–1)
    """
    dish_ings = dish_row['ingredient_list']
    dish_qtys = parse_quantities(dish_row['quantities_g'])
    base_servings = dish_row.get('servings', 4)

    if not dish_qtys or len(dish_qtys) != len(dish_ings):
        max_servings = base_servings
    else:
        ratios = []
        for ing, req_qty_base in zip(dish_ings, dish_qtys):
            ing_clean = ing.lower().replace(' ', '_')
            avail = inventory.get(ing_clean, 0)
            if req_qty_base > 0:
                possible = (avail / req_qty_base) * base_servings
                ratios.append(possible)

        max_servings = int(min(ratios)) if ratios else base_servings
        max_servings = max(0, max_servings)

    demand_alignment = min(max_servings / max(predicted_meals, 1), 1.0)

    return {
        'max_servings': max_servings,
        'demand_alignment': round(demand_alignment, 4)
    }

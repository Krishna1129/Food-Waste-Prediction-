"""
Train Recommender - CLI entry point for the Dish Recommendation Engine.

Usage:
    python src/train_recommender.py

This script:
1. Loads or auto-generates the dishes dataset and nutrition data
2. Builds the dish feature matrix
3. Generates synthetic ranking labels
4. Trains an XGBoost ranker
5. Saves the model to models/dish_recommender_model.pkl
6. Prints evaluation summary and runs a quick demo
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from src.dish_dataset import (
    load_or_generate_dishes,
    load_or_generate_nutrition,
    compute_dish_features,
)
from src.inventory import (
    parse_inventory,
    compute_batch_match_scores,
    compute_expiry_risk,
    compute_dish_waste_reduction_score,
    estimate_servings,
    parse_quantities,
)
from src.recommend import (
    build_ranking_features,
    generate_synthetic_ranking_labels,
    RANKER_FEATURE_COLS,
    save_recommender,
    recommend_dishes,
    explain_dish_ranking,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

# A representative inventory used to compute training-time ranking features
TRAINING_INVENTORY = {
    "rice": 50, "wheat": 30, "lentils": 40, "potato": 25,
    "tomato": 20, "onion": 20, "garlic": 10, "ginger": 10,
    "oil": 10, "turmeric": 5, "cumin": 5, "coriander": 5,
    "paneer": 15, "yogurt": 20, "butter": 10, "cream": 8,
    "peas": 15, "carrot": 15, "cauliflower": 15, "spinach": 20,
    "egg": 10, "chicken": 20, "mushroom": 10, "capsicum": 10,
    "chickpea": 15, "kidney_bean": 10, "flour": 20,
    "pasta": 15, "noodles": 15, "bread": 10, "semolina": 10,
    "flattened_rice": 10, "soy_sauce": 5, "garam_masala": 5,
}

TRAINING_PREDICTED_MEALS = 300


def train(
    dishes_dataset_path: str = None,
    nutrition_dataset_path: str = None,
    output_path: str = None
) -> dict:
    """
    Full training pipeline for the dish recommender.

    Returns:
        dict: saved model artifacts
    """
    print("\n" + "=" * 60)
    print("🍽️  DISH RECOMMENDER - MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Load datasets ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1️⃣  LOADING / GENERATING DATASETS")
    print("=" * 60)

    dishes_raw = load_or_generate_dishes(dishes_dataset_path)
    nutrition_df = load_or_generate_nutrition(nutrition_dataset_path)

    # ── 2. Compute dish features ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2️⃣  COMPUTING DISH FEATURES")
    print("=" * 60)

    dishes_feat = compute_dish_features(dishes_raw)
    print(f"  ✅ {len(dishes_feat)} dishes with {len(dishes_feat.columns)} features")

    # ── 3. Compute inventory match scores ────────────────────────────────
    print("\n" + "=" * 60)
    print("3️⃣  COMPUTING INVENTORY MATCH SCORES")
    print("=" * 60)

    inv = parse_inventory(TRAINING_INVENTORY)
    dishes_scored = compute_batch_match_scores(dishes_feat, inv)

    avg_usage = dishes_scored['usage_score'].mean()
    print(f"  ✅ Average inventory usage score: {avg_usage:.2%}")

    # ── 4. Build ranking features ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4️⃣  BUILDING RANKING FEATURE MATRIX")
    print("=" * 60)

    dishes_enriched = build_ranking_features(
        dishes_scored, inv, TRAINING_PREDICTED_MEALS, nutrition_df
    )

    valid_mask = dishes_enriched[RANKER_FEATURE_COLS].notna().all(axis=1)
    dishes_clean = dishes_enriched[valid_mask].reset_index(drop=True)

    print(f"  ✅ {len(dishes_clean)} / {len(dishes_enriched)} dishes after NaN removal")
    print(f"  ✅ Feature columns: {RANKER_FEATURE_COLS}")

    X = dishes_clean[RANKER_FEATURE_COLS].values.astype(float)

    # ── 5. Generate synthetic labels ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("5️⃣  GENERATING SYNTHETIC RANKING LABELS")
    print("=" * 60)

    y = generate_synthetic_ranking_labels(dishes_clean)

    print(f"  ✅ Label range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  ✅ Mean label : {y.mean():.4f}")

    # ── 6. Train XGBoost Model ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6️⃣  TRAINING XGBOOST RANKER")
    print("=" * 60)
    print(f"  📊 Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"  ⚙️  Params : {config.XGBOOST_RANKER_PARAMS}")

    model = xgb.XGBRegressor(**config.XGBOOST_RANKER_PARAMS)
    model.fit(X, y)

    train_preds = model.predict(X)
    train_corr = float(np.corrcoef(y, train_preds)[0, 1])
    from sklearn.metrics import mean_absolute_error, r2_score
    train_mae = mean_absolute_error(y, train_preds)
    train_r2 = r2_score(y, train_preds)

    print(f"\n  📊 Training Metrics (ranking proxy):")
    print(f"     Pearson Correlation : {train_corr:.4f}")
    print(f"     MAE                 : {train_mae:.4f}")
    print(f"     R²                  : {train_r2:.4f}")

    # ── 7. Save model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7️⃣  SAVING MODEL")
    print("=" * 60)

    artifacts = {
        'model': model,
        'feature_cols': RANKER_FEATURE_COLS,
        'candidate_dishes': dishes_clean.to_dict('records'),
        'dishes_df': dishes_clean,
        'training_metrics': {
            'pearson_corr': train_corr,
            'mae': train_mae,
            'r2': train_r2,
        },
        'training_inventory': TRAINING_INVENTORY,
        'training_predicted_meals': TRAINING_PREDICTED_MEALS,
    }

    save_recommender(artifacts, output_path)

    return artifacts


def evaluate_demo(artifacts: dict):
    """Run a quick demo to validate the trained recommender end-to-end."""
    print("\n" + "="*60)
    print("8️⃣  QUICK DEMO - recommend_dishes()")
    print("="*60)

    test_inventory = {
        "rice": 50, "potato": 20, "onion": 10,
        "tomato": 15, "oil": 5, "paneer": 8,
        "lentils": 30, "turmeric": 2, "cumin": 2,
        "garlic": 5, "ginger": 5, "peas": 10
    }

    print("\n📦 Test Inventory:")
    for k, v in test_inventory.items():
        print(f"   {k}: {v}g")

    print("\n🔄 Getting recommendations (menu_type='veg', predicted_meals=200)...")
    results = recommend_dishes(
        inventory=test_inventory,
        predicted_meals=200,
        menu_type='veg',
        top_n=5
    )

    print(f"\n🏆 Top {len(results)} Recommended Dishes:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['dish_name']} ({r['menu_type']}, {r['cuisine']})")
        print(f"     Usage Score     : {r['inventory_usage_score']:.2%}")
        print(f"     Waste Reduction : {r['waste_reduction_score']:.2%}")
        print(f"     Est. Servings   : {r['estimated_servings']}")
        print(f"     Confidence      : {r['confidence_score']:.4f}")
        print(f"     Calories/Serving: {r['calories_per_serving']} kcal")
        if r['missing_ingredients']:
            missing_str = ', '.join(r['missing_ingredients'][:4])
            print(f"     Missing (top 4) : {missing_str}")

    # ── SHAP explanation for top dish ─────────────────────────────────────
    if results:
        print("\n" + "="*60)
        print("9️⃣  SHAP EXPLANATION FOR TOP DISH")
        print("="*60)
        top_dish = results[0]['dish_name']
        explain_dish_ranking(top_dish, artifacts)


def main():
    """Entry point for `python src/train_recommender.py`."""
    print("\n" + "="*60)
    print("🚀 FOOD WASTE PREDICTION - DISH RECOMMENDER TRAINING")
    print("="*60)

    try:
        artifacts = train()
        evaluate_demo(artifacts)

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📌 Next steps:")
        print(f"   ▶  Run recommendations : python src/recommend.py")
        print(f"   ▶  Check model artifact : {config.RECOMMENDER_MODEL_FILE}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

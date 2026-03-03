"""
Configuration settings for the Food Waste Prediction System.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# Data files
DATA_FILE = os.path.join(DATA_DIR, 'food_waste_prediction_large_dataset_10y.csv')
MODEL_FILE = os.path.join(MODELS_DIR, 'food_waste_model.pkl')

# Recommender data files
DISHES_DATASET_FILE = os.path.join(DATA_DIR, 'dishes_dataset.csv')
INGREDIENT_NUTRITION_FILE = os.path.join(DATA_DIR, 'ingredient_nutrition.csv')
RECOMMENDER_MODEL_FILE = os.path.join(MODELS_DIR, 'dish_recommender_model.pkl')

# Model parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# Feature lists
NUMERICAL_FEATURES = [
    'occupancy_rate',  # Changed from hostel_occupancy_rate to support both hostels and restaurants
    'temperature_c',
    'prev_day_meals',
    'prev_7day_avg_meals',
    'meals_prepared'
]

CATEGORICAL_FEATURES = [
    'weather',
    'menu_type',
    'facility_type'  # NEW: distinguish between hostel and restaurant
]

# Model hyperparameters
LINEAR_REGRESSION_PARAMS = {}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Visualization settings
VIZ_DPI = 300
VIZ_FIGSIZE = (18, 5)
SHAP_TOP_N = 10

# ─────────────────────────────────────────────
# RECOMMENDER SETTINGS
# ─────────────────────────────────────────────

# Minimum fraction of dish ingredients that must be in inventory (0-1)
MIN_INGREDIENT_MATCH = 0.40

# Default top-N recommendations
RECOMMENDER_TOP_N = 5

# Random state for synthetic data generation
RECOMMENDER_RANDOM_STATE = 42

# XGBoost ranker hyperparameters
XGBOOST_RANKER_PARAMS = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RECOMMENDER_RANDOM_STATE,
    'n_jobs': -1
}

# Valid menu types
MENU_TYPES = ['veg', 'non-veg', 'vegan', 'any']

# Common allergen groups (ingredients to watch)
ALLERGEN_GROUPS = {
    'gluten':   ['wheat', 'flour', 'maida', 'semolina', 'rava', 'bread', 'pasta', 'noodles'],
    'dairy':    ['milk', 'paneer', 'cheese', 'butter', 'curd', 'cream', 'ghee', 'yogurt'],
    'nuts':     ['peanut', 'cashew', 'almond', 'walnut', 'pistachio', 'groundnut'],
    'eggs':     ['egg', 'eggs'],
    'soy':      ['soy', 'tofu', 'tempeh']
}

# SHAP top-N for recommender explanations
RECOMMENDER_SHAP_TOP_N = 8

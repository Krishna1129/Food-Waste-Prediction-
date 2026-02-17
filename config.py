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

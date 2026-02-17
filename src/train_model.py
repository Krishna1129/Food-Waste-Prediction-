"""
Model training module for the Food Waste Prediction System.

This script trains multiple models, compares them, and saves the best one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    load_data, explore_data, preprocess_data,
    split_data, create_preprocessing_pipeline
)
from src.utils import get_feature_names
import config


def train_models(X_train, y_train, preprocessing_pipeline):
    """
    Train multiple models and compare their performance.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessing_pipeline: Preprocessing pipeline
        
    Returns:
        tuple: (best_model, best_model_name, results, preprocessing_pipeline, feature_names)
    """
    print("\n" + "="*60)
    print("🚀 TRAINING MULTIPLE MODELS")
    print("="*60)
    
    # Fit preprocessing pipeline
    print("\n🔄 Fitting preprocessing pipeline...")
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    print(f"✅ Preprocessing complete. Transformed shape: {X_train_transformed.shape}")
    
    # Get feature names
    feature_names = get_feature_names(
        preprocessing_pipeline,
        config.CATEGORICAL_FEATURES,
        config.NUMERICAL_FEATURES
    )
    print(f"✅ Total features after encoding: {len(feature_names)}")
    
    # Create validation split
    X_train_fit, X_val, y_train_fit, y_val = sklearn_split(
        X_train_transformed, y_train,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=False
    )
    
    print(f"\n📊 Training samples: {len(X_train_fit)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    # Dictionary to store models and results
    models = {}
    results = {}
    
    print("\n" + "="*60)
    print("🔄 TRAINING MODELS")
    print("="*60)
    
    # 1. Linear Regression
    print("\n1️⃣  Training Linear Regression...")
    lr_model = LinearRegression(**config.LINEAR_REGRESSION_PARAMS)
    lr_model.fit(X_train_fit, y_train_fit)
    models['Linear Regression'] = lr_model
    print("   ✅ Linear Regression trained")
    
    # 2. Random Forest
    print("\n2️⃣  Training Random Forest...")
    rf_model = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)
    rf_model.fit(X_train_fit, y_train_fit)
    models['Random Forest'] = rf_model
    print("   ✅ Random Forest trained")
    
    # 3. XGBoost
    print("\n3️⃣  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    xgb_model.fit(X_train_fit, y_train_fit)
    models['XGBoost'] = xgb_model
    print("   ✅ XGBoost trained")
    
    # Evaluate all models
    print("\n" + "="*60)
    print("📊 EVALUATING MODELS ON VALIDATION SET")
    print("="*60)
    
    for model_name, model in models.items():
        y_val_pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)
        
        results[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'model': model
        }
    
    # Display comparison table
    display_comparison_table(results)
    
    # Select best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['MAE'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
    print("="*60)
    print(f"   MAE:  {results[best_model_name]['MAE']:.2f} meals")
    print(f"   RMSE: {results[best_model_name]['RMSE']:.2f} meals")
    print(f"   R²:   {results[best_model_name]['R2']:.4f}")
    
    return best_model, best_model_name, results, preprocessing_pipeline, feature_names


def display_comparison_table(results):
    """Display model comparison table."""
    print("\n┌─────────────────────┬──────────┬──────────┬──────────┐")
    print("│ Model               │   MAE    │   RMSE   │    R²    │")
    print("├─────────────────────┼──────────┼──────────┼──────────┤")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])
    
    for idx, (model_name, metrics) in enumerate(sorted_results):
        medal = "🥇 " if idx == 0 else "   "
        print(f"│ {medal}{model_name:17s} │ {metrics['MAE']:8.2f} │ {metrics['RMSE']:8.2f} │ {metrics['R2']:8.4f} │")
    
    print("└─────────────────────┴──────────┴──────────┴──────────┘")


def evaluate_model(model, preprocessing_pipeline, X_train, y_train, X_test, y_test):
    """
    Evaluate model on train and test sets.
    
    Args:
        model: Trained model
        preprocessing_pipeline: Preprocessing pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print("📈 EVALUATING MODEL")
    print("="*60)
    
    # Transform data
    X_train_transformed = preprocessing_pipeline.transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)
    
    # Predictions
    y_train_pred = model.predict(X_train_transformed)
    y_test_pred = model.predict(X_test_transformed)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n📊 Training Set Performance:")
    print(f"   MAE:  {train_mae:.2f} meals")
    print(f"   RMSE: {train_rmse:.2f} meals")
    print(f"   R²:   {train_r2:.4f}")
    
    print("\n📊 Test Set Performance:")
    print(f"   MAE:  {test_mae:.2f} meals")
    print(f"   RMSE: {test_rmse:.2f} meals")
    print(f"   R²:   {test_r2:.4f}")
    
    metrics = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
    
    return metrics


def save_model(model, best_model_name, results, preprocessing_pipeline, 
               feature_names, filepath=None):
    """
    Save trained model and artifacts.
    
    Args:
        model: Trained model
        best_model_name (str): Name of best model
        results (dict): Model comparison results
        preprocessing_pipeline: Preprocessing pipeline
        feature_names (list): Feature names
        filepath (str): Path to save model
    """
    if filepath is None:
        filepath = config.MODEL_FILE
    
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_artifacts = {
        'model': model,
        'best_model_name': best_model_name,
        'model_comparison_results': results,
        'preprocessing_pipeline': preprocessing_pipeline,
        'feature_names': feature_names,
        'categorical_features': config.CATEGORICAL_FEATURES,
        'numerical_features': config.NUMERICAL_FEATURES
    }
    
    joblib.dump(model_artifacts, filepath)
    print(f"✅ Model saved to: {filepath}")
    print(f"✅ Best model: {best_model_name}")
    print(f"✅ Preprocessing pipeline saved")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🍽️  FOOD WASTE PREDICTION - MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and preprocess data
    df = load_data()
    explore_data(df)
    X, y, df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, df_processed)
    
    # Create preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline()
    
    # Train models
    best_model, best_model_name, results, preprocessing_pipeline, feature_names = train_models(
        X_train, y_train, preprocessing_pipeline
    )
    
    # Evaluate best model
    metrics = evaluate_model(
        best_model, preprocessing_pipeline,
        X_train, y_train, X_test, y_test
    )
    
    # Save model
    save_model(
        best_model, best_model_name, results,
        preprocessing_pipeline, feature_names
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return best_model, metrics


if __name__ == '__main__':
    main()

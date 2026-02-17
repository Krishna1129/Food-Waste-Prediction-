"""
Prediction module for the Food Waste Prediction System.

This module provides functions to load a trained model and make predictions.
"""

import pandas as pd
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import DateFeatureExtractor, AdvancedFeatureEngineer
import config


def load_model(filepath=None):
    """
    Load a trained model and its artifacts.
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        dict: Model artifacts
    """
    if filepath is None:
        filepath = config.MODEL_FILE
    
    print(f"📂 Loading model from: {filepath}")
    model_artifacts = joblib.load(filepath)
    
    print("✅ Model loaded successfully!")
    print(f"✅ Model type: {model_artifacts.get('best_model_name', 'Unknown')}")
    print("✅ Preprocessing pipeline loaded")
    
    return model_artifacts


def predict_meals(input_data, model_artifacts):
    """
    Predict meals served for a single day.
    
    Args:
        input_data (dict): Dictionary containing input features
        model_artifacts (dict): Loaded model artifacts
        
    Returns:
        float: Predicted number of meals
        
    Example:
        >>> model_artifacts = load_model()
        >>> input_data = {
        ...     'date': '2026-03-15',
        ...     'hostel_occupancy_rate': 0.85,
        ...     'temperature_c': 28.5,
        ...     'is_weekend': 0,
        ...     'is_holiday': 0,
        ...     'event_flag': 0,
        ...     'exam_period': 1,
        ...     'prev_day_meals': 450,
        ...     'prev_7day_avg_meals': 445,
        ...     'meals_prepared': 500,
        ...     'weather': 'clear',
        ...     'menu_type': 'standard_veg',
        ...     'day_of_week': 5,
        ...     'meals_served': 450
        ... }
        >>> prediction = predict_meals(input_data, model_artifacts)
        >>> print(f"Predicted: {prediction:.0f} meals")
    """
    # Extract artifacts
    model = model_artifacts['model']
    preprocessing_pipeline = model_artifacts['preprocessing_pipeline']
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply date feature extraction
    date_extractor = DateFeatureExtractor(date_column='date')
    df = date_extractor.transform(df)
    
    # Apply advanced feature engineering
    advanced_engineer = AdvancedFeatureEngineer(
        target_col='meals_served',
        create_lags=True,
        create_rolling=True,
        create_trend=True,
        create_interactions=True
    )
    df = advanced_engineer.transform(df)
    
    # Select features
    numerical_features = model_artifacts['numerical_features']
    categorical_features = model_artifacts['categorical_features']
    all_features = numerical_features + categorical_features
    X = df[all_features]
    
    # Transform using preprocessing pipeline
    X_transformed = preprocessing_pipeline.transform(X)
    
    # Make prediction
    prediction = model.predict(X_transformed)[0]
    
    return prediction


def predict_batch(input_data_list, model_artifacts):
    """
    Predict meals for multiple days.
    
    Args:
        input_data_list (list): List of input dictionaries
        model_artifacts (dict): Loaded model artifacts
        
    Returns:
        list: List of predictions
    """
    predictions = [predict_meals(data, model_artifacts) for data in input_data_list]
    return predictions


def main():
    """Example usage of prediction module."""
    print("\n" + "="*60)
    print("🔮 FOOD WASTE PREDICTION - INFERENCE")
    print("="*60)
    
    # Load model
    model_artifacts = load_model()
    
    # Example input
    input_data = {
        'date': '2026-03-15',
        'occupancy_rate': 0.85,
        'temperature_c': 28.5,
        'is_weekend': 0,
        'is_holiday': 0,
        'event_flag': 0,
        'exam_period': 1,
        'prev_day_meals': 450,
        'prev_7day_avg_meals': 445,
        'meals_prepared': 500,
        'weather': 'clear',
        'menu_type': 'standard_veg',
        'facility_type': 'hostel',
        'day_of_week': 5,
        'meals_served': 450
    }
    
    print("\n📋 Input Data:")
    print(f"   Date: {input_data['date']}")
    print(f"   Occupancy: {input_data['occupancy_rate']*100:.0f}%")
    print(f"   Facility Type: {input_data['facility_type'].capitalize()}")
    print(f"   Temperature: {input_data['temperature_c']}°C")
    print(f"   Exam Period: {'Yes' if input_data['exam_period'] else 'No'}")
    
    # Make prediction
    prediction = predict_meals(input_data, model_artifacts)
    
    print(f"\n🎯 Prediction: {prediction:.0f} meals")
    print("\n✅ Prediction completed!")


if __name__ == '__main__':
    main()

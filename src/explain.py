"""
Explainability module for the Food Waste Prediction System.

This module uses SHAP to explain model predictions.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, predict_meals
from src.utils import DateFeatureExtractor, AdvancedFeatureEngineer
import config


def explain_prediction(input_data, model_artifacts, top_n=10, show_plot=True):
    """
    Explain a prediction using SHAP values.
    
    Args:
        input_data (dict): Input features
        model_artifacts (dict): Loaded model artifacts
        top_n (int): Number of top features to display
        show_plot (bool): Whether to show waterfall plot
        
    Returns:
        dict: Explanation with prediction, base value, and contributions
    """
    print("\n" + "="*60)
    print("🔍 EXPLAINING PREDICTION")
    print("="*60)
    
    # Make prediction first
    prediction = predict_meals(input_data, model_artifacts)
    
    # Extract artifacts
    model = model_artifacts['model']
    preprocessing_pipeline = model_artifacts['preprocessing_pipeline']
    feature_names = model_artifacts['feature_names']
    numerical_features = model_artifacts['numerical_features']
    categorical_features = model_artifacts['categorical_features']
    best_model_name = model_artifacts.get('best_model_name', 'Unknown')
    
    # Prepare data for SHAP
    df = pd.DataFrame([input_data])
    
    # Apply transformations
    date_extractor = DateFeatureExtractor(date_column='date')
    df = date_extractor.transform(df)
    
    advanced_engineer = AdvancedFeatureEngineer(
        target_col='meals_served',
        create_lags=True,
        create_rolling=True,
        create_trend=True,
        create_interactions=True
    )
    df = advanced_engineer.transform(df)
    
    # Select features and transform
    all_features = numerical_features + categorical_features
    X = df[all_features]
    X_transformed = preprocessing_pipeline.transform(X)
    
    # Create SHAP explainer
    print("\n🔄 Computing SHAP values...")
    
    if 'Linear' in best_model_name:
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    
    # Get base value
    base_value = explainer.expected_value
    
    # Get SHAP values for single sample
    if len(shap_values.shape) > 1:
        sample_shap_values = shap_values[0]
    else:
        sample_shap_values = shap_values
    
    # Create feature contribution dataframe
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': sample_shap_values,
        'Abs SHAP': np.abs(sample_shap_values)
    }).sort_values('Abs SHAP', ascending=False)
    
    # Display results
    print("\n" + "="*60)
    print(f"📊 PREDICTION EXPLANATION")
    print("="*60)
    print(f"\n🎯 Predicted Meals: {prediction:.0f}")
    print(f"📍 Base Value (Average): {base_value:.0f} meals")
    print(f"📈 Total Impact: {prediction - base_value:+.0f} meals")
    
    print(f"\n🔝 Top {top_n} Features Influencing This Prediction:")
    print("-" * 60)
    
    # Display top features with explanations
    for idx, (_, row) in enumerate(contributions.head(top_n).iterrows(), 1):
        feature = row['Feature']
        shap_val = row['SHAP Value']
        
        if shap_val > 0:
            direction = "INCREASES"
            arrow = "⬆️"
            sign = "+"
        else:
            direction = "DECREASES"
            arrow = "⬇️"
            sign = ""
        
        explanation = get_feature_explanation(feature, shap_val, input_data)
        
        print(f"\n{idx}. {arrow} {feature}")
        print(f"   Impact: {sign}{shap_val:.2f} meals ({direction} prediction)")
        print(f"   💡 {explanation}")
    
    # Show SHAP waterfall plot
    if show_plot:
        print("\n📊 Generating SHAP waterfall plot...")
        
        # Ensure visualizations directory exists
        os.makedirs(config.VIZ_DIR, exist_ok=True)
        
        shap_explanation = shap.Explanation(
            values=sample_shap_values,
            base_values=base_value,
            data=X_transformed[0],
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, max_display=top_n, show=False)
        plt.title(f'SHAP Waterfall Plot - Top {top_n} Features\nPrediction: {prediction:.0f} meals',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        model_name_clean = best_model_name.lower().replace(" ", "_")
        filename = os.path.join(config.VIZ_DIR, f'shap_explanation_{model_name_clean}.png')
        plt.savefig(filename, dpi=config.VIZ_DPI, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved SHAP plot: {filename}")
        plt.show()
    
    # Return explanation
    explanation = {
        'prediction': prediction,
        'base_value': base_value,
        'total_impact': prediction - base_value,
        'top_features': contributions.head(top_n).to_dict('records')
    }
    
    print("\n✅ Explanation completed!")
    
    return explanation


def get_feature_explanation(feature_name, shap_value, input_data):
    """Generate simple language explanation for a feature."""
    direction = "increases" if shap_value > 0 else "decreases"
    magnitude = abs(shap_value)
    
    if 'lag_1_day' in feature_name:
        return f"Yesterday's meal count strongly {direction} today's prediction"
    elif 'lag_7_day' in feature_name:
        return f"Last week's meal count {direction} the prediction"
    elif 'rolling_mean' in feature_name:
        days = '7' if '7' in feature_name else '14'
        return f"The {days}-day average trend {direction} the prediction"
    elif 'rolling_std' in feature_name:
        return f"Recent variability in meals {direction} the prediction"
    elif 'trend' in feature_name:
        return f"The recent trend pattern {direction} the prediction"
    elif 'occupancy' in feature_name.lower():
        if any(x in feature_name for x in ['weekend', 'holiday', 'exam', 'event']):
            return f"Combined effect of occupancy with special conditions {direction} prediction"
        else:
            occ = input_data.get('hostel_occupancy_rate', 0)
            return f"Hostel occupancy of {occ*100:.0f}% {direction} the prediction"
    elif 'temperature' in feature_name.lower():
        if 'weekend' in feature_name:
            return f"Weekend temperature effect {direction} the prediction"
        else:
            temp = input_data.get('temperature_c', 0)
            return f"Temperature of {temp:.1f}°C {direction} the prediction"
    elif 'weekend' in feature_name.lower():
        return f"Weekend status {direction} the prediction"
    elif 'holiday' in feature_name.lower():
        return f"Holiday status {direction} the prediction"
    elif 'exam' in feature_name.lower():
        return f"Exam period {direction} the prediction"
    elif 'event' in feature_name.lower():
        return f"Special event {direction} the prediction"
    elif 'meals_prepared' in feature_name:
        meals_prep = input_data.get('meals_prepared', 0)
        return f"Preparing {meals_prep:.0f} meals {direction} the prediction"
    elif 'weather' in feature_name:
        return f"Weather conditions {direction} the prediction"
    elif 'menu' in feature_name:
        return f"Menu type {direction} the prediction"
    elif 'day' in feature_name or 'month' in feature_name:
        return f"Time of year/week {direction} the prediction"
    else:
        return f"This feature {direction} the prediction by {magnitude:.1f} meals"


def main():
    """Example usage of explainability module."""
    print("\n" + "="*60)
    print("🔍 FOOD WASTE PREDICTION - EXPLAINABILITY")
    print("="*60)
    
    # Load model
    model_artifacts = load_model()
    
    # Example input
    input_data = {
        'date': '2026-03-15',
        'hostel_occupancy_rate': 0.92,
        'temperature_c': 27.5,
        'is_weekend': 0,
        'is_holiday': 0,
        'event_flag': 0,
        'exam_period': 1,
        'prev_day_meals': 480,
        'prev_7day_avg_meals': 475,
        'meals_prepared': 520,
        'weather': 'clear',
        'menu_type': 'standard_veg',
        'day_of_week': 5,
        'meals_served': 480
    }
    
    print("\n📋 Input Scenario:")
    print(f"   Date: {input_data['date']}")
    print(f"   Occupancy: {input_data['hostel_occupancy_rate']*100:.0f}%")
    print(f"   Exam Period: Yes")
    
    # Explain prediction
    explanation = explain_prediction(input_data, model_artifacts, top_n=config.SHAP_TOP_N)
    
    print("\n" + "="*60)
    print("✅ Explainability demo completed!")
    print("="*60)


if __name__ == '__main__':
    main()

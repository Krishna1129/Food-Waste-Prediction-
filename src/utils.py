"""
Utility classes and transformers for the Food Waste Prediction System.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract date-based features from a date column."""
    
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])
        
        # Extract basic date features
        X['day_of_week'] = X[self.date_column].dt.dayofweek
        X['day_of_month'] = X[self.date_column].dt.day
        X['month'] = X[self.date_column].dt.month
        X['quarter'] = X[self.date_column].dt.quarter
        X['week_of_year'] = X[self.date_column].dt.isocalendar().week
        
        # Cyclical encoding for day of week
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        
        # Cyclical encoding for month
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create advanced time-series and interaction features."""
    
    def __init__(self, target_col='meals_served', create_lags=True,
                 create_rolling=True, create_trend=True, create_interactions=True):
        self.target_col = target_col
        self.create_lags = create_lags
        self.create_rolling = create_rolling
        self.create_trend = create_trend
        self.create_interactions = create_interactions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Lag features
        if self.create_lags and self.target_col in X.columns:
            X['lag_1_day'] = X[self.target_col].shift(1)
            X['lag_7_day'] = X[self.target_col].shift(7)
            X['lag_14_day'] = X[self.target_col].shift(14)
        
        # Rolling statistics
        if self.create_rolling and self.target_col in X.columns:
            X['rolling_mean_7'] = X[self.target_col].shift(1).rolling(window=7, min_periods=1).mean()
            X['rolling_mean_14'] = X[self.target_col].shift(1).rolling(window=14, min_periods=1).mean()
            X['rolling_std_7'] = X[self.target_col].shift(1).rolling(window=7, min_periods=1).std()
            X['rolling_min_7'] = X[self.target_col].shift(1).rolling(window=7, min_periods=1).min()
            X['rolling_max_7'] = X[self.target_col].shift(1).rolling(window=7, min_periods=1).max()
        
        # Trend features
        if self.create_trend and self.target_col in X.columns:
            X['trend_7_day'] = X[self.target_col].shift(1) - X[self.target_col].shift(8)
            X['trend_14_day'] = X[self.target_col].shift(1) - X[self.target_col].shift(15)
            X['pct_change_7'] = X[self.target_col].shift(1).pct_change(periods=7, fill_method=None)
        
        # Interaction features
        if self.create_interactions:
            if 'is_weekend' in X.columns and 'occupancy_rate' in X.columns:
                X['weekend_occupancy'] = X['is_weekend'] * X['occupancy_rate']
            
            if 'is_weekend' in X.columns and 'temperature_c' in X.columns:
                X['weekend_temp'] = X['is_weekend'] * X['temperature_c']
            
            if 'is_holiday' in X.columns and 'occupancy_rate' in X.columns:
                X['holiday_occupancy'] = X['is_holiday'] * X['occupancy_rate']
            
            if 'exam_period' in X.columns and 'occupancy_rate' in X.columns:
                X['exam_occupancy'] = X['exam_period'] * X['occupancy_rate']
            
            if 'event_flag' in X.columns and 'occupancy_rate' in X.columns:
                X['event_occupancy'] = X['event_flag'] * X['occupancy_rate']
        
        return X


def get_feature_names(preprocessing_pipeline, categorical_features, numerical_features):
    """Extract feature names after preprocessing."""
    feature_names = []
    
    # Numerical features (scaled)
    feature_names.extend(numerical_features)
    
    # Categorical features (one-hot encoded)
    if hasattr(preprocessing_pipeline, 'named_transformers_'):
        cat_encoder = preprocessing_pipeline.named_transformers_['cat']
        if hasattr(cat_encoder, 'named_steps'):
            onehot = cat_encoder.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                cat_feature_names = onehot.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
    
    return feature_names

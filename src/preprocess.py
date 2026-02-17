"""
Data preprocessing module for the Food Waste Prediction System.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import DateFeatureExtractor, DataFrameSelector, AdvancedFeatureEngineer
import config


def load_data(filepath=None):
    """
    Load the food waste dataset.
    
    Args:
        filepath (str): Path to CSV file. If None, uses config.DATA_FILE
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if filepath is None:
        filepath = config.DATA_FILE
    
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Backward compatibility: rename hostel_occupancy_rate to occupancy_rate
    if 'hostel_occupancy_rate' in df.columns:
        print(f"  ℹ️  Renaming 'hostel_occupancy_rate' to 'occupancy_rate' for multi-facility support")
        df.rename(columns={'hostel_occupancy_rate': 'occupancy_rate'}, inplace=True)
    
    # Add facility_type if missing (for old datasets, default to 'hostel')
    if 'facility_type' not in df.columns:
        print(f"  ℹ️  Adding 'facility_type' column (defaulting to 'hostel')")
        df['facility_type'] = 'hostel'
    
    return df


def explore_data(df):
    """
    Display basic statistics about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nTarget Variable (meals_served):")
    print(f"  Mean: {df['meals_served'].mean():.2f}")
    print(f"  Std: {df['meals_served'].std():.2f}")
    print(f"  Min: {df['meals_served'].min():.2f}")
    print(f"  Max: {df['meals_served'].max():.2f}")
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  None")
    else:
        print(missing[missing > 0])


def preprocess_data(df):
    """
    Apply feature engineering and prepare data for modeling.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y, df_processed)
    """
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DATA")
    print("="*60)
    
    df = df.copy()
    
    # Extract date features
    print("\n1️⃣  Extracting date features...")
    date_extractor = DateFeatureExtractor(date_column='date')
    df = date_extractor.transform(df)
    print("   ✅ Date features extracted")
    
    # Create advanced features
    print("\n2️⃣  Creating advanced features...")
    advanced_engineer = AdvancedFeatureEngineer(
        target_col='meals_served',
        create_lags=True,
        create_rolling=True,
        create_trend=True,
        create_interactions=True
    )
    df = advanced_engineer.transform(df)
    print("   ✅ Advanced features created")
    
    # Drop rows with NaN (from lag features)
    print("\n3️⃣  Handling missing values...")
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    print(f"   ✅ Dropped {dropped_rows} rows with missing values")
    print(f"   ✅ Remaining: {len(df)} rows")
    
    # Separate features and target
    y = df['meals_served']
    X = df.drop('meals_served', axis=1)
    
    print(f"\n✅ Preprocessing completed!")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X, y, df


def split_data(X, y, df, test_size=None):
    """
    Split data into train and test sets (time-based split).
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        df (pd.DataFrame): Full dataset with date
        test_size (float): Proportion for test set
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    
    print("\n" + "="*60)
    print("✂️  SPLITTING DATA")
    print("="*60)
    
    # Time-based split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=config.RANDOM_STATE
    )
    
    # Get date ranges
    train_dates = df.loc[X_train.index, 'date']
    test_dates = df.loc[X_test.index, 'date']
    
    print(f"\n📅 Training Set:")
    print(f"   Size: {len(X_train)} samples")
    print(f"   Date Range: {train_dates.min()} to {train_dates.max()}")
    
    print(f"\n📅 Test Set:")
    print(f"   Size: {len(X_test)} samples")
    print(f"   Date Range: {test_dates.min()} to {test_dates.max()}")
    
    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(numerical_features=None, categorical_features=None):
    """
    Create sklearn preprocessing pipeline.
    
    Args:
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    if numerical_features is None:
        numerical_features = config.NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = config.CATEGORICAL_FEATURES
    
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(numerical_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_features)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combined pipeline
    preprocessing_pipeline = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    
    return preprocessing_pipeline


if __name__ == '__main__':
    # Test preprocessing
    df = load_data()
    explore_data(df)
    X, y, df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, df_processed)
    
    print("\n✅ Preprocessing module test completed!")

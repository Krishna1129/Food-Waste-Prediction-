# Food Waste Prediction System

A production-ready machine learning system for predicting daily food waste in hostel cafeterias.

## � Live Demo
Experience the application live: [**Food Waste Prediction System**](https://food-waste-prediction-rakh.onrender.com)

## �📁 Project Structure

```
reckon/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── src/                     # Source code
│   ├── utils.py            # Custom transformers and utilities
│   ├── preprocess.py       # Data preprocessing
│   ├── train_model.py      # Model training
│   ├── predict.py          # Prediction/inference
│   └── explain.py          # SHAP explainability
│
├── data/                    # Data files
│   └── food_waste_prediction_large_dataset_10y.csv
│
├── models/                  # Saved models
│   └── food_waste_model.pkl
│
└── visualizations/          # Generated plots
    ├── model_performance_*.png
    └── shap_explanation_*.png
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python src/train_model.py
```

This will:
- Load and preprocess data
- Train 3 models (Linear Regression, Random Forest, XGBoost)
- Compare performance and select the best
- Save the best model to `models/food_waste_model.pkl`

### 3. Make Predictions

```python
from src.predict import load_model, predict_meals

# Load trained model
model_artifacts = load_model()

# Prepare input
input_data = {
    'date': '2026-03-15',
    'hostel_occupancy_rate': 0.85,
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
    'day_of_week': 5,
    'meals_served': 450
}

# Predict
prediction = predict_meals(input_data, model_artifacts)
print(f"Predicted meals: {prediction:.0f}")
```

### 4. Explain Predictions

```python
from src.explain import load_model, explain_prediction

# Load model
model_artifacts = load_model()

# Explain prediction
explanation = explain_prediction(input_data, model_artifacts, top_n=10)
```


## 🌍 Deployment

For detailed instructions on deploying this application to Render (recommended), please refer to [DEPLOYMENT.md](DEPLOYMENT.md).

## 📊 Features

### Data Preprocessing
- **Date Features**: Day of week, month, cyclical encoding
- **Lag Features**: 1-day, 7-day, 14-day lags
- **Rolling Statistics**: 7-day and 14-day means, std, min, max
- **Trend Features**: 7-day and 14-day trends, percentage changes
- **Interaction Features**: Weekend×occupancy, exam×occupancy, etc.

### Model Training
- **Multiple Models**: Linear Regression, Random Forest, XGBoost
- **Automatic Selection**: Best model chosen based on validation MAE
- **Comprehensive Metrics**: MAE, RMSE, R² scores

### Explainability
- **SHAP Values**: Feature contribution analysis
- **Simple Explanations**: Plain language interpretations
- **Waterfall Plots**: Visual feature importance

## 📝 Module Documentation

### config.py
Central configuration for:
- File paths
- Model hyperparameters
- Feature lists
- Visualization settings

### src/utils.py
Custom transformers:
- `DateFeatureExtractor`: Extract date-based features
- `AdvancedFeatureEngineer`: Create lag, rolling, and interaction features
- `DataFrameSelector`: Select specific columns

### src/preprocess.py
Data preprocessing functions:
- `load_data()`: Load CSV dataset
- `explore_data()`: Display statistics
- `preprocess_data()`: Apply feature engineering
- `split_data()`: Time-based train/test split
- `create_preprocessing_pipeline()`: Build sklearn pipeline

### src/train_model.py
Model training functions:
- `train_models()`: Train and compare multiple models
- `evaluate_model()`: Calculate performance metrics
- `save_model()`: Save model and artifacts

### src/predict.py
Prediction functions:
- `load_model()`: Load trained model
- `predict_meals()`: Predict for single day
- `predict_batch()`: Predict for multiple days

### src/explain.py
Explainability functions:
- `explain_prediction()`: SHAP-based explanation
- `get_feature_explanation()`: Generate simple explanations

## 🎯 Model Performance

Expected performance (varies by data):

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **XGBoost** | 15-25 | 20-30 | 0.92-0.96 |
| Random Forest | 18-30 | 24-35 | 0.88-0.94 |
| Linear Regression | 40-60 | 50-75 | 0.65-0.80 |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    ...
}

# Data split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Features
NUMERICAL_FEATURES = [...]
CATEGORICAL_FEATURES = [...]
```

## 📈 Example Workflow

```python
# 1. Train model
from src.train_model import main as train_main
train_main()

# 2. Load and predict
from src.predict import load_model, predict_meals
model_artifacts = load_model()
prediction = predict_meals(input_data, model_artifacts)

# 3. Explain prediction
from src.explain import explain_prediction
explanation = explain_prediction(input_data, model_artifacts)
```

## 🐛 Troubleshooting

**Import errors:**
```bash
# Ensure you're in the project root directory
cd /path/to/reckon
python src/train_model.py
```

**Missing data:**
```bash
# Move data file to data/ directory
mv food_waste_prediction_large_dataset_10y.csv data/
```

**Model not found:**
```bash
# Train model first
python src/train_model.py
```

## 📦 Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- shap >= 0.43.0
- joblib >= 1.3.0

## 🎓 Key Concepts

**Time-Series Split**: Data is split chronologically (no shuffling) to prevent data leakage.

**Feature Engineering**: Advanced features capture temporal patterns and interactions.

**Model Selection**: Multiple models are trained and the best is automatically selected.

**Explainability**: SHAP values show which features drive each prediction.

## 📄 License

MIT License

## 👥 Authors

Food Waste Prediction System Team

---

**Built with ❤️ for sustainable food management**

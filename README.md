# 🍽️ Smart Food Waste Prediction System

> **AI-powered meal demand forecasting + dish recommendation engine for hostels & restaurants.**  
> Minimise food waste, match inventory to demand, and get intelligent dish suggestions — all from one web interface.

---

## 🌐 Live Demo

**[food-waste-prediction-rakh.onrender.com](https://food-waste-prediction-rakh.onrender.com)**

---

## ✨ What It Does

| Feature | Description |
|---|---|
| 🎯 **Meal Demand Predictor** | Forecasts exact meals served for any day using XGBoost / Random Forest / Linear Regression |
| 🥘 **Dish Recommender** | Recommends dishes from kitchen inventory to maximise usage & minimise waste |
| 🔍 **SHAP Explainability** | Plain-language + waterfall-plot explanations for every prediction |
| 🌐 **Web Interface** | Beautiful two-tab Flask UI — no code required |

---

## 📁 Project Structure

```
reckon/
├── app.py                          # Flask web app (predict + recommend API)
├── config.py                       # Central configuration
├── requirements.txt
├── Dockerfile / DEPLOYMENT.md
│
├── src/
│   ├── train_model.py              # Train food-waste prediction model
│   ├── predict.py                  # Meal demand inference
│   ├── preprocess.py               # Data loading & feature engineering
│   ├── explain.py                  # SHAP explainability
│   ├── utils.py                    # Custom sklearn transformers
│   ├── dish_dataset.py             # Recipe dataset loader & feature builder
│   ├── inventory.py                # Inventory parser, match scorer, expiry risk
│   ├── recommend.py                # Hybrid dish recommendation engine + SHAP
│   └── train_recommender.py        # Train dish recommender model (CLI)
│
├── data/
│   ├── food_waste_prediction_large_dataset_10y.csv
│   ├── dishes_dataset.csv          # 74-dish recipe dataset (auto-generated)
│   └── ingredient_nutrition.csv    # Nutrition + shelf-life table (auto-generated)
│
├── models/
│   ├── food_waste_model.pkl        # Trained prediction model
│   └── dish_recommender_model.pkl  # Trained XGBoost ranker
│
├── static/css/style.css
├── static/js/app.js
└── templates/index.html
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the meal demand model

```bash
python src/train_model.py
```

Trains Linear Regression, Random Forest & XGBoost, auto-selects the best, saves to `models/food_waste_model.pkl`.

### 3. Train the dish recommender

```bash
python src/train_recommender.py
```

Auto-generates the recipe & nutrition datasets if missing, trains an XGBoost ranker, saves to `models/dish_recommender_model.pkl`.

### 4. Run the web app

```bash
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 🎯 Tab 1 — Meal Demand Predictor

Enter daily parameters and get an instant AI forecast:

```python
from src.predict import load_model, predict_meals

model_artifacts = load_model()

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
    'facility_type': 'hostel',   # or 'restaurant'
    'day_of_week': 5,
    'meals_served': 450
}

prediction = predict_meals(input_data, model_artifacts)
print(f"Predicted meals: {prediction:.0f}")
```

### SHAP Explanation

```python
from src.explain import explain_prediction

explanation = explain_prediction(input_data, model_artifacts, top_n=10)
```

---

## 🥘 Tab 2 — Dish Recommender from Inventory

Given what's in the kitchen, rank the best dishes to cook:

```python
from src.recommend import recommend_dishes

inventory = {
    "rice": 50,      # grams
    "potato": 20,
    "onion": 10,
    "tomato": 15,
    "oil": 5,
    "paneer": 8,
    "lentils": 30,
    "garlic": 5
}

results = recommend_dishes(
    inventory=inventory,
    predicted_meals=200,          # from Tab 1 output or manual entry
    menu_type='veg',              # 'veg' | 'non-veg' | 'vegan' | 'any'
    top_n=5,
    allergens=['nuts'],           # optional exclusion list
    days_until_expiry={'paneer': 2}  # optional — boosts near-expiry items
)
```

Each result contains:

```python
{
    "dish_name": "Dal Tadka",
    "inventory_usage_score": 0.81,   # fraction of ingredients available
    "waste_reduction_score": 0.37,   # urgency of near-expiry items used
    "estimated_servings": 12,
    "confidence_score": 0.78,        # ML ranker score
    "menu_type": "veg",
    "cuisine": "indian",
    "calories_per_serving": 230,
    "missing_ingredients": ["coriander"],
    "prep_time_min": 40
}
```

### SHAP Explanation for a Dish

```python
from src.recommend import load_recommender, explain_dish_ranking

artifacts = load_recommender()
explain_dish_ranking("Dal Tadka", artifacts)
```

---

## 📊 Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost** *(auto-selected)* | 15–25 | 20–30 | 0.92–0.96 |
| Random Forest | 18–30 | 24–35 | 0.88–0.94 |
| Linear Regression | 40–60 | 50–75 | 0.65–0.80 |

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `MIN_INGREDIENT_MATCH` | `0.40` | Min fraction of dish ingredients required |
| `RECOMMENDER_TOP_N` | `5` | Default recommendations returned |
| `XGBOOST_RANKER_PARAMS` | see file | Ranker hyperparameters |
| `ALLERGEN_GROUPS` | gluten/dairy/nuts/eggs/soy | Ingredient exclusion groups |
| `MENU_TYPES` | veg/non-veg/vegan/any | Valid menu filter options |

---

## 🔌 API Endpoints

### `POST /predict`

```json
{
  "date": "2026-03-15",
  "occupancy": 85,
  "temperature": 26,
  "is_weekend": "false",
  "is_holiday": "false",
  "exam_period": "true",
  "event_flag": "false",
  "prev_day_meals": 450,
  "prev_7day_avg": 445,
  "meals_prepared": 480,
  "weather": "clear",
  "menu_type": "standard_veg",
  "facility_type": "hostel",
  "day_of_week": 3
}
```

**Response:** `{ "success": true, "prediction": 463 }`

---

### `POST /recommend`

```json
{
  "inventory": { "rice": 50, "potato": 20, "paneer": 8 },
  "predicted_meals": 200,
  "menu_type": "veg",
  "top_n": 5,
  "allergens": []
}
```

**Response:** `{ "success": true, "recommendations": [ ... ] }`

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: food_waste_model.pkl` | Run `python src/train_model.py` |
| `FileNotFoundError: dish_recommender_model.pkl` | Run `python src/train_recommender.py` |
| Import errors | Run from project root: `python app.py` |
| Empty recommendations | Lower `MIN_INGREDIENT_MATCH` in `config.py` or switch menu type to `any` |

---

## 📦 Dependencies

```
pandas==2.1.4        numpy==1.26.3        scikit-learn==1.3.2
xgboost==2.0.3       shap==0.44.1         matplotlib==3.8.2
joblib==1.3.2        flask==3.0.3         gunicorn==21.2.0
```

---

## 🌍 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Render deployment instructions.

```bash
# Docker
docker build -t food-waste-system .
docker run -p 5000:5000 food-waste-system
```

---

## 📄 License

MIT License

---

**Built with ❤️ for sustainable, intelligent food management**

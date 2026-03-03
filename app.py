"""
Flask Web Application for Food Waste Prediction System

A beautiful, interactive web interface for making predictions
and dish recommendations from kitchen inventory.
"""

from flask import Flask, render_template, request, jsonify
from src.predict import load_model, predict_meals
import os

app = Flask(__name__)

# ── Load food-waste prediction model once at startup ──────────────────────
print("Loading food-waste model...")
model_artifacts = load_model()
print("Model loaded successfully!")

# ── Lazy-load recommender (done on first /recommend call) ─────────────────
_recommender_artifacts = None

def get_recommender():
    global _recommender_artifacts
    if _recommender_artifacts is None:
        try:
            from src.recommend import load_recommender
            _recommender_artifacts = load_recommender()
        except FileNotFoundError:
            _recommender_artifacts = None
    return _recommender_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle meal-count prediction requests."""
    try:
        data = request.get_json()
        print("\n" + "="*60)
        print("📥 Received prediction request")
        print("="*60)

        input_data = {
            'date': data.get('date'),
            'occupancy_rate': float(data.get('occupancy')) / 100,
            'temperature_c': float(data.get('temperature')),
            'is_weekend': 1 if data.get('is_weekend') == 'true' else 0,
            'is_holiday': 1 if data.get('is_holiday') == 'true' else 0,
            'event_flag': 1 if data.get('event_flag') == 'true' else 0,
            'exam_period': 1 if data.get('exam_period') == 'true' else 0,
            'prev_day_meals': int(data.get('prev_day_meals')),
            'prev_7day_avg_meals': int(data.get('prev_7day_avg')),
            'meals_prepared': int(data.get('meals_prepared')),
            'weather': data.get('weather'),
            'menu_type': data.get('menu_type'),
            'facility_type': data.get('facility_type', 'hostel'),
            'day_of_week': int(data.get('day_of_week')),
            'meals_served': int(data.get('prev_day_meals'))
        }

        print(f"\n📊 Occupancy: {input_data['occupancy_rate']*100}%  "
              f"Temp: {input_data['temperature_c']}°C")

        prediction = predict_meals(input_data, model_artifacts)
        print(f"✅ Prediction: {prediction:.0f} meals")

        return jsonify({'success': True, 'prediction': round(prediction, 0)})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': f"{type(e).__name__}: {str(e)}"}), 400


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle dish recommendation requests.

    Expected JSON body:
    {
        "inventory":        { "rice": 50, "potato": 20, ... },
        "predicted_meals":  200,
        "menu_type":        "veg",     // veg | non-veg | vegan | any
        "top_n":            5,
        "allergens":        []         // optional
    }
    """
    try:
        data = request.get_json()
        print("\n" + "="*60)
        print("🍽️  Received recommendation request")
        print("="*60)

        inventory       = data.get('inventory', {})
        predicted_meals = int(data.get('predicted_meals', 200))
        menu_type       = data.get('menu_type', 'any')
        top_n           = int(data.get('top_n', 5))
        allergens       = data.get('allergens', []) or []

        if not inventory:
            return jsonify({'success': False, 'error': 'Inventory is empty.'}), 400

        print(f"  Inventory items : {len(inventory)}")
        print(f"  Predicted meals : {predicted_meals}")
        print(f"  Menu type       : {menu_type}")

        from src.recommend import recommend_dishes
        results = recommend_dishes(
            inventory=inventory,
            predicted_meals=predicted_meals,
            menu_type=menu_type,
            top_n=top_n,
            allergens=allergens if allergens else None,
        )

        print(f"✅ Returning {len(results)} recommendations")
        return jsonify({'success': True, 'recommendations': results})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': f"{type(e).__name__}: {str(e)}"}), 400


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🍽️  FOOD WASTE PREDICTION - WEB INTERFACE")
    print("="*60)
    print("\n🌐 Starting server...")
    print("📱 Open your browser: http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)


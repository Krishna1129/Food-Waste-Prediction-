"""
Flask Web Application for Food Waste Prediction System

A beautiful, interactive web interface for making predictions.
"""

from flask import Flask, render_template, request, jsonify
from src.predict import load_model, predict_meals
import os

app = Flask(__name__)

# Load model once at startup
print("Loading model...")
model_artifacts = load_model()
print("Model loaded successfully!")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        data = request.get_json()
        print("\n" + "="*60)
        print("📥 Received prediction request")
        print("="*60)
        print(f"Raw data: {data}")
        
        # Prepare input
        input_data = {
            'date': data.get('date'),
            'occupancy_rate': float(data.get('occupancy')) / 100,  # Changed from hostel_occupancy_rate
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
            'facility_type': data.get('facility_type', 'hostel'),  # NEW: facility type
            'day_of_week': int(data.get('day_of_week')),
            'meals_served': int(data.get('prev_day_meals'))  # Use prev_day for features
        }
        
        print(f"\n📊 Processed input data:")
        print(f"   Occupancy: {input_data['occupancy_rate']*100}%")
        print(f"   Temperature: {input_data['temperature_c']}°C")
        print(f"   Exam Period: {input_data['exam_period']}")
        print(f"   Previous Day: {input_data['prev_day_meals']}")
        
        # Make prediction
        print(f"\n🔄 Making prediction...")
        prediction = predict_meals(input_data, model_artifacts)
        print(f"✅ Prediction: {prediction:.0f} meals")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 0)
        })
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}"
        }), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🍽️  FOOD WASTE PREDICTION - WEB INTERFACE")
    print("="*60)
    print("\n🌐 Starting server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    # Get port from environment variable for Render deployment, default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

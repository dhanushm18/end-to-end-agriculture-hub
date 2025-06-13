from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from crop_data import CROP_DATA, MARKET_TRENDS, REGIONAL_SUITABILITY
    print("Crop data loaded successfully!")
except ImportError as e:
    print(f"Error importing crop data: {e}")
    # Fallback minimal data
    CROP_DATA = {}
    MARKET_TRENDS = {}
    REGIONAL_SUITABILITY = {}

app = Flask(__name__)

# render home page
@app.route('/')
def home():
    title = 'AGRI HUB - AI Powered Crop & Soil Management'
    return render_template('index.html', title=title)

# render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'AGRI HUB - Smart Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AGRI HUB - Soil Optimizer'
    return render_template('fertilizer.html', title=title)

# render disease prediction page
@app.route('/disease-predict')
def disease_prediction():
    title = 'AGRI HUB - Disease Detection'
    return render_template('disease.html', title=title)

# render chatbot page
@app.route('/chatbot')
def chatbot():
    title = 'AGRI HUB - AI Agriculture Assistant'
    return render_template('chatbot.html', title=title)

# render weather recommendations page
@app.route('/weather-recommendations')
def weather_recommendations():
    title = 'AGRI HUB - Weather-Aware Recommendations'
    return render_template('weather_recommendations.html', title=title)

# render crop comparison dashboard
@app.route('/crop-comparison')
def crop_comparison():
    title = 'AGRI HUB - Crop Comparison Dashboard'
    return render_template('crop_comparison.html', title=title)

# crop comparison API endpoints
@app.route('/crop-data-api', methods=['GET'])
def crop_data_api():
    """Get all crop data for comparison"""
    try:
        return {
            'success': True,
            'crops': CROP_DATA,
            'market_trends': MARKET_TRENDS,
            'regional_suitability': REGIONAL_SUITABILITY
        }
    except Exception as e:
        print(f"Crop data API error: {e}")
        return {'success': False, 'error': 'Unable to fetch crop data'}

@app.route('/crop-compare-api', methods=['POST'])
def crop_compare_api():
    """Compare selected crops"""
    try:
        data = request.get_json()
        selected_crops = data.get('crops', [])
        
        if not selected_crops:
            return {'success': False, 'error': 'Please select at least one crop'}
        
        comparison_data = {}
        for crop in selected_crops:
            if crop in CROP_DATA:
                comparison_data[crop] = CROP_DATA[crop]
        
        return {
            'success': True,
            'comparison': comparison_data,
            'market_trends': {crop: MARKET_TRENDS.get(crop, {}) for crop in selected_crops},
            'regional_data': {crop: REGIONAL_SUITABILITY.get(crop, []) for crop in selected_crops}
        }
    except Exception as e:
        print(f"Crop comparison API error: {e}")
        return {'success': False, 'error': 'Unable to compare crops'}

if __name__ == '__main__':
    print("Starting AGRI HUB Crop Comparison Test Server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

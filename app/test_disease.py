from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Try to import disease dictionary
try:
    from utils.disease import disease_dic
    print("Disease dictionary loaded successfully!")
except ImportError as e:
    print(f"Error importing disease dictionary: {e}")
    disease_dic = {}

# Mock disease classes for testing
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy', 'Tomato___healthy']

def predict_image_mock(img):
    """Mock prediction function for testing"""
    # Return a sample prediction for testing
    return 'Apple___healthy'

@app.route('/')
def home():
    title = 'AGRI HUB - AI Powered Crop & Soil Management'
    return render_template('index.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AGRI HUB - Disease Detection AI'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        
        try:
            # Read and validate the uploaded file
            if file.filename == '':
                error_message = "Please select a valid image file."
                return render_template('disease.html', title=title, error=error_message)
            
            # Check file extension
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                error_message = "Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP)."
                return render_template('disease.html', title=title, error=error_message)
            
            img = file.read()
            
            # Get prediction from mock function
            prediction = predict_image_mock(img)
            
            # Check if prediction is valid
            if prediction in disease_dic:
                prediction_text = Markup(str(disease_dic[prediction]))
                return render_template('disease-result.html', prediction=prediction_text, title=title)
            else:
                # For testing, show a default message
                prediction_text = Markup(f"""
                <b>Crop</b>: Apple <br/>Disease: No disease detected<br/>
                <br/><br/> 
                <b>Good news!</b> Your plant appears to be healthy. Keep up the good care!
                <br/><br/>
                <b>General Tips:</b>
                <br/>• Continue regular watering and fertilization
                <br/>• Monitor for any changes in leaf color or texture
                <br/>• Ensure proper sunlight and air circulation
                <br/>• Remove any dead or damaged leaves promptly
                """)
                return render_template('disease-result.html', prediction=prediction_text, title=title)
                
        except Exception as e:
            print(f"Disease prediction error: {e}")
            error_message = "An error occurred while processing your image. Please try again with a different image."
            return render_template('disease.html', title=title, error=error_message)
    
    return render_template('disease.html', title=title)

# Add other basic routes for navigation
@app.route('/crop-recommend')
def crop_recommend():
    title = 'AGRI HUB - Smart Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AGRI HUB - Soil Optimizer'
    return render_template('fertilizer.html', title=title)

@app.route('/chatbot')
def chatbot():
    title = 'AGRI HUB - AI Agriculture Assistant'
    return render_template('chatbot.html', title=title)

@app.route('/weather-recommendations')
def weather_recommendations():
    title = 'AGRI HUB - Weather-Aware Recommendations'
    return render_template('weather_recommendations.html', title=title)

@app.route('/crop-comparison')
def crop_comparison():
    title = 'AGRI HUB - Crop Comparison Dashboard'
    return render_template('crop_comparison.html', title=title)

if __name__ == '__main__':
    print("Starting AGRI HUB Disease Test Server...")
    print(f"Disease dictionary has {len(disease_dic)} entries")
    app.run(debug=True, host='0.0.0.0', port=5000)

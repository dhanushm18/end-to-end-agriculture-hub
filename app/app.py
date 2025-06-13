# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import json
from crop_data import CROP_DATA, MARKET_TRENDS, REGIONAL_SUITABILITY
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model
print("Loading disease classification model...")

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

print(f"Total disease classes: {len(disease_classes)}")

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model
print("Loading crop recommendation model...")
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
print("Crop recommendation model loaded successfully!")

# Simple Chatbot Configuration
print("Configuring simple agriculture chatbot...")
print("✅ Simple agriculture chatbot ready!")

# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    try:
        api_key = config.weather_api_key
        base_url = "http://api.openweathermap.org/data/2.5/weather?"

        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()

        print(f"Weather API response for {city_name}: {x}")  # Debug output

        # Check if the response is successful
        if response.status_code == 200 and x.get("cod") == 200 and "main" in x:
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            print(f"Weather data for {city_name}: Temperature={temperature}°C, Humidity={humidity}%")
            return temperature, humidity
        else:
            print(f"Weather API error for {city_name}: {x}")
            return None
    except Exception as e:
        print(f"Weather fetch error for {city_name}: {e}")
        return None




def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def get_chatbot_response(user_message):
    """
    Get response from Google Gemini AI for agriculture-related queries
    :params: user_message
    :return: chatbot response
    """
    try:
        # Create agriculture-focused prompt for Gemini
        agriculture_prompt = f"""You are AGRI BOT, an expert agricultural AI assistant for the AGRI HUB platform. You specialize in:

- Crop recommendations and farming techniques
- Plant disease identification and treatment
- Soil management and fertilizer advice
- Weather-based agricultural planning
- Irrigation and water management
- Pest control and organic farming
- Sustainable agriculture practices

Please provide helpful, accurate, and practical agricultural advice. Keep responses informative but concise (2-4 sentences).

User question: {user_message}

Response:"""

        # Use Gemini API with working model
        api_key = config.google_ai_api_key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            "contents": [{
                "parts": [{
                    "text": agriculture_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 500,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }

        print(f"Sending request to Gemini API for: {user_message[:50]}...")
        response = requests.post(url, headers=headers, json=data, timeout=15)

        print(f"Gemini API response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"API Response structure: {list(result.keys())}")

            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                print(f"Candidate structure: {list(candidate.keys())}")

                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        ai_response = parts[0]['text'].strip()
                        print(f"✅ Gemini response received: {ai_response[:100]}...")
                        return ai_response

            print("❌ Unexpected response structure from Gemini")
            return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."

        else:
            print(f"❌ Gemini API error: {response.status_code} - {response.text}")
            return "I'm sorry, I'm experiencing technical difficulties connecting to the AI service. Please try again later."

    except requests.exceptions.Timeout:
        print("❌ Gemini API timeout")
        return "I'm taking longer than usual to respond. Please try asking your question again."

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return "I'm having trouble connecting to the AI service. Please check your internet connection and try again."

    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return "I apologize, but I encountered an unexpected error. Please try rephrasing your question."




def get_weather_forecast(city):
    """
    Get 5-day weather forecast for a city
    :params: city
    :return: weather forecast data
    """
    try:
        api_key = config.weather_api_key
        # Get current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        current_response = requests.get(current_url)

        if current_response.status_code == 200:
            current_data = current_response.json()

            # Get 5-day forecast (free API)
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
            forecast_response = requests.get(forecast_url)

            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()

                # Process forecast data to get daily summaries
                daily_forecast = []
                current_date = None
                daily_data = {}

                for item in forecast_data['list']:
                    date = item['dt_txt'].split(' ')[0]

                    if date != current_date:
                        if current_date and daily_data:
                            daily_forecast.append(daily_data)

                        current_date = date
                        daily_data = {
                            'dt': item['dt'],
                            'temp': {'max': item['main']['temp'], 'min': item['main']['temp']},
                            'humidity': item['main']['humidity'],
                            'weather': item['weather'],
                            'wind_speed': item['wind']['speed']
                        }
                    else:
                        # Update min/max temperatures
                        daily_data['temp']['max'] = max(daily_data['temp']['max'], item['main']['temp'])
                        daily_data['temp']['min'] = min(daily_data['temp']['min'], item['main']['temp'])
                        daily_data['humidity'] = (daily_data['humidity'] + item['main']['humidity']) / 2

                # Add the last day
                if daily_data:
                    daily_forecast.append(daily_data)

                return {
                    'success': True,
                    'city': city,
                    'current': current_data,
                    'forecast': {'daily': daily_forecast[:7]}  # Limit to 7 days
                }
            else:
                return {'success': False, 'error': 'Unable to fetch forecast data'}
        else:
            return {'success': False, 'error': 'City not found'}
    except Exception as e:
        print(f"Weather API error: {e}")
        return {'success': False, 'error': 'Weather service unavailable'}


def generate_weather_recommendations(weather_data):
    """
    Generate comprehensive agricultural recommendations based on weather forecast
    :params: weather_data
    :return: recommendations with activities, alerts, and farming guidance
    """
    try:
        if not weather_data['success']:
            return []

        recommendations = []
        daily_forecast = weather_data['forecast']['daily'][:7]  # 7 days

        for i, day in enumerate(daily_forecast):
            day_name = ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'][i]
            temp_max = day['temp']['max']
            temp_min = day['temp']['min']
            humidity = day['humidity']
            weather_main = day['weather'][0]['main']
            weather_desc = day['weather'][0]['description']
            wind_speed = day.get('wind_speed', 0)

            # Initialize recommendation categories
            planting_activities = []
            harvesting_activities = []
            general_recommendations = []
            alerts = []
            irrigation_advice = []

            # Determine farming activities based on weather conditions
            is_good_for_planting = True
            is_good_for_harvesting = True
            is_good_for_spraying = True

            # Temperature analysis
            if temp_max > 35:
                alerts.append("🔥 HEAT ALERT: Extreme temperature expected")
                general_recommendations.append("🌡️ High temperature - ensure adequate irrigation and shade for crops")
                irrigation_advice.append("💧 Increase watering frequency, preferably early morning (5-7 AM) or evening (6-8 PM)")
                is_good_for_planting = False
                planting_activities.append("❌ Avoid planting - high heat stress risk")
            elif temp_max > 30:
                general_recommendations.append("🌞 Warm weather - monitor soil moisture levels")
                irrigation_advice.append("💧 Regular watering needed - check soil moisture daily")
            elif temp_max < 5:
                alerts.append("❄️ FROST ALERT: Freezing temperatures expected")
                general_recommendations.append("🛡️ Protect sensitive crops with row covers or greenhouse protection")
                is_good_for_planting = False
                is_good_for_harvesting = False
                planting_activities.append("❌ No planting - frost damage risk")
                harvesting_activities.append("❌ Delay harvesting - frost may damage crops")
            elif temp_max < 10:
                alerts.append("🌡️ COLD WARNING: Very low temperatures")
                general_recommendations.append("❄️ Cold weather - protect sensitive crops from cold damage")
                is_good_for_planting = False
                planting_activities.append("⚠️ Limited planting - only cold-resistant varieties")

            # Humidity analysis
            if humidity > 85:
                alerts.append("💨 HIGH HUMIDITY ALERT: Disease risk increased")
                general_recommendations.append("🍄 Monitor for fungal diseases - improve air circulation")
                general_recommendations.append("� Apply preventive fungicide if necessary")
                is_good_for_spraying = False
            elif humidity < 25:
                alerts.append("🌵 LOW HUMIDITY WARNING: Drought stress risk")
                irrigation_advice.append("💧 Increase irrigation frequency and consider mulching")

            # Weather condition analysis
            if weather_main == 'Rain':
                if 'heavy' in weather_desc.lower() or 'thunderstorm' in weather_desc.lower():
                    alerts.append("⛈️ SEVERE WEATHER ALERT: Heavy rain/storms expected")
                    general_recommendations.append("🚜 Avoid all field operations - soil compaction risk")
                    general_recommendations.append("⚠️ Ensure proper drainage to prevent waterlogging")
                    is_good_for_planting = False
                    is_good_for_harvesting = False
                    is_good_for_spraying = False
                    planting_activities.append("❌ No planting - waterlogging risk")
                    harvesting_activities.append("❌ Postpone harvesting - crop damage risk")
                else:
                    general_recommendations.append("🌧️ Light rain expected - natural irrigation")
                    irrigation_advice.append("💧 Reduce or skip irrigation - natural watering")
                    is_good_for_spraying = False
                    planting_activities.append("⚠️ Limited planting - wait for soil to drain")

            elif weather_main == 'Clear':
                general_recommendations.append("☀️ Clear skies - excellent for most farm activities")
                if temp_max < 30 and temp_min > 10:
                    planting_activities.append("✅ EXCELLENT for planting - ideal conditions")
                    harvesting_activities.append("✅ PERFECT for harvesting - dry conditions")
                else:
                    planting_activities.append("✅ Good for planting with proper timing")
                    harvesting_activities.append("✅ Good for harvesting")

            elif weather_main == 'Clouds':
                general_recommendations.append("☁️ Cloudy conditions - reduced evaporation")
                irrigation_advice.append("💧 Adjust irrigation - lower water loss")
                planting_activities.append("✅ Good for planting - reduced heat stress")
                harvesting_activities.append("✅ Suitable for harvesting")

            elif weather_main in ['Thunderstorm', 'Storm']:
                alerts.append("⛈️ STORM WARNING: Severe weather approaching")
                general_recommendations.append("🏠 Secure equipment and protect crops")
                general_recommendations.append("🐄 Move livestock to shelter")
                is_good_for_planting = False
                is_good_for_harvesting = False
                is_good_for_spraying = False
                planting_activities.append("❌ NO outdoor activities - storm danger")
                harvesting_activities.append("❌ Postpone all harvesting - safety risk")

            # Wind analysis
            if wind_speed > 10:
                alerts.append("💨 WIND ALERT: High winds expected")
                general_recommendations.append("🌪️ Secure loose materials and support tall plants")
                is_good_for_spraying = False

            # Spraying recommendations
            if is_good_for_spraying and weather_main == 'Clear' and wind_speed < 5:
                general_recommendations.append("� IDEAL for pesticide/fertilizer spraying")
            elif not is_good_for_spraying:
                general_recommendations.append("❌ Avoid spraying - weather conditions not suitable")

            # Temperature variation analysis
            if temp_max - temp_min > 15:
                alerts.append("🌡️ TEMPERATURE STRESS: Large day-night variation")
                general_recommendations.append("📊 Monitor crops for temperature stress")

            # Specific crop activities based on conditions
            if is_good_for_planting:
                if temp_max < 25 and temp_min > 15:
                    planting_activities.append("🌱 Ideal for cool-season crops (lettuce, spinach, peas)")
                elif temp_max < 30 and temp_min > 18:
                    planting_activities.append("🌽 Good for warm-season crops (tomatoes, peppers, corn)")

            if is_good_for_harvesting:
                if humidity < 60 and weather_main == 'Clear':
                    harvesting_activities.append("🌾 Perfect for grain harvesting - low moisture")
                    harvesting_activities.append("🍅 Excellent for fruit/vegetable harvesting")
                elif weather_main == 'Clear':
                    harvesting_activities.append("✅ Good harvesting conditions")

            # Irrigation timing
            if weather_main != 'Rain':
                if temp_max > 25:
                    irrigation_advice.append("⏰ Best irrigation time: Early morning (5-7 AM) or evening (6-8 PM)")
                else:
                    irrigation_advice.append("⏰ Flexible irrigation timing - moderate temperatures")

            # Compile all recommendations
            all_recommendations = []

            if alerts:
                all_recommendations.extend(alerts)
            if planting_activities:
                all_recommendations.append("� PLANTING ACTIVITIES:")
                all_recommendations.extend([f"   {activity}" for activity in planting_activities])
            if harvesting_activities:
                all_recommendations.append("🌾 HARVESTING ACTIVITIES:")
                all_recommendations.extend([f"   {activity}" for activity in harvesting_activities])
            if irrigation_advice:
                all_recommendations.append("💧 IRRIGATION GUIDANCE:")
                all_recommendations.extend([f"   {advice}" for advice in irrigation_advice])
            if general_recommendations:
                all_recommendations.append("📋 GENERAL RECOMMENDATIONS:")
                all_recommendations.extend([f"   {rec}" for rec in general_recommendations])

            recommendations.append({
                'day': day_name,
                'date': day,
                'recommendations': all_recommendations,
                'activity_summary': {
                    'planting': 'Excellent' if is_good_for_planting and temp_max < 30 else 'Good' if is_good_for_planting else 'Not Recommended',
                    'harvesting': 'Excellent' if is_good_for_harvesting and weather_main == 'Clear' else 'Good' if is_good_for_harvesting else 'Not Recommended',
                    'spraying': 'Excellent' if is_good_for_spraying and weather_main == 'Clear' else 'Good' if is_good_for_spraying else 'Not Recommended',
                    'irrigation': 'Not Needed' if weather_main == 'Rain' else 'High Priority' if temp_max > 30 else 'Normal'
                },
                'alerts_count': len(alerts)
            })

        return recommendations
    except Exception as e:
        print(f"Recommendation generation error: {e}")
        return []

def generate_weather_summary(weather_data):
    """
    Generate summary of weather recommendations instead of daily details
    :params: weather_data
    :return: summary recommendations with key insights
    """
    try:
        if not weather_data['success']:
            return {'summary': [], 'weekly_outlook': {}}

        daily_forecast = weather_data['forecast']['daily'][:7]  # 7 days

        # Analyze overall week patterns
        total_rain_days = 0
        total_hot_days = 0
        total_cold_days = 0
        total_high_humidity_days = 0

        critical_alerts = []
        planting_opportunities = []
        harvesting_opportunities = []
        irrigation_needs = []
        general_advice = []

        # Analyze each day for patterns
        for i, day in enumerate(daily_forecast):
            day_name = ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'][i]
            temp_max = day['temp']['max']
            temp_min = day['temp']['min']
            humidity = day['humidity']
            weather_main = day['weather'][0]['main']
            weather_desc = day['weather'][0]['description']
            wind_speed = day.get('wind_speed', 0)

            # Count weather patterns
            if weather_main == 'Rain':
                total_rain_days += 1
            if temp_max > 35:
                total_hot_days += 1
            if temp_max < 5:
                total_cold_days += 1
            if humidity > 85:
                total_high_humidity_days += 1

            # Identify critical alerts
            if temp_max > 35:
                critical_alerts.append(f"🔥 {day_name}: Extreme heat warning - Temperature reaching {temp_max:.1f}°C. Crops at risk of heat stress and dehydration.")
            elif temp_max < 5:
                critical_alerts.append(f"❄️ {day_name}: Frost alert - Temperature dropping to {temp_max:.1f}°C. Immediate crop protection required.")

            if weather_main in ['Thunderstorm', 'Storm'] or 'heavy' in weather_desc.lower():
                critical_alerts.append(f"⛈️ {day_name}: Severe weather warning - {weather_desc.title()}. Avoid all outdoor farm operations for safety.")

            # Identify opportunities
            if weather_main == 'Clear' and temp_max < 30 and temp_min > 10 and wind_speed < 5:
                planting_opportunities.append(f"✅ {day_name}: Ideal planting conditions")
                harvesting_opportunities.append(f"✅ {day_name}: Perfect harvesting weather")

        # Generate summary recommendations
        summary_recommendations = []

        # Weather pattern summary
        if total_rain_days > 3:
            summary_recommendations.append("Wet Week Alert: Expect frequent rainfall throughout the week. Plan indoor farm activities and ensure proper field drainage to prevent waterlogging.")
            irrigation_needs.append("Reduce irrigation frequency as natural rainfall will provide adequate water for crops.")
        elif total_rain_days == 0:
            summary_recommendations.append("Dry Week Alert: No rainfall expected this week. Increase irrigation frequency and monitor soil moisture levels closely.")
            irrigation_needs.append("High irrigation priority - water crops daily, especially during hot periods.")
        else:
            summary_recommendations.append(f"Mixed Weather: Expect {total_rain_days} rainy days this week. Plan farm activities around dry periods.")
            irrigation_needs.append("Moderate irrigation needed - adjust watering schedule based on daily rainfall.")

        if total_hot_days > 2:
            summary_recommendations.append(f"Heat Wave Warning: {total_hot_days} extremely hot days (above 35°C) expected. Protect crops from heat stress and increase watering frequency.")
            irrigation_needs.append("Critical irrigation timing: Water crops early morning (5-7 AM) and evening (6-8 PM) to minimize evaporation.")

        if total_cold_days > 0:
            summary_recommendations.append(f"Cold Weather Alert: {total_cold_days} days with freezing temperatures expected. Use row covers, mulching, or greenhouse protection for sensitive crops.")

        if total_high_humidity_days > 3:
            summary_recommendations.append(f"Disease Risk Alert: {total_high_humidity_days} days with high humidity (above 85%) expected. Monitor crops for fungal diseases and improve air circulation around plants.")

        # Best days summary
        if planting_opportunities:
            best_planting = ', '.join([day.split(':')[0].replace('✅ ', '') for day in planting_opportunities[:3]])
            summary_recommendations.append(f"Optimal Planting Window: Best days for planting are {best_planting}. Clear skies and moderate temperatures provide ideal conditions.")
        else:
            summary_recommendations.append("Limited Planting Opportunities: Weather conditions are not ideal for planting this week. Consider postponing planting activities.")

        if harvesting_opportunities:
            best_harvesting = ', '.join([day.split(':')[0].replace('✅ ', '') for day in harvesting_opportunities[:3]])
            summary_recommendations.append(f"Prime Harvesting Days: Best days for harvesting are {best_harvesting}. Dry conditions will ensure quality crop collection.")
        else:
            summary_recommendations.append("Challenging Harvesting Conditions: Weather may affect harvesting operations. Plan accordingly and protect harvested crops.")

        # General weekly advice
        clear_calm_days = len([day for day in daily_forecast if day['weather'][0]['main'] == 'Clear' and day.get('wind_speed', 0) < 5])
        if clear_calm_days > 2:
            general_advice.append(f"Excellent Spraying Conditions: {clear_calm_days} days with clear, calm weather ideal for pesticide and fertilizer application.")
        else:
            general_advice.append("Limited Spraying Opportunities: Weather conditions not suitable for spraying. Wait for calmer, clearer days.")

        # Irrigation timing advice
        if total_hot_days > 0:
            irrigation_needs.append("Best Watering Schedule: Water crops during cooler hours - early morning (5-7 AM) or evening (6-8 PM) for maximum efficiency.")

        # Compile final summary with better formatting
        final_summary = []

        if critical_alerts:
            final_summary.append("🚨 CRITICAL WEATHER ALERTS")
            final_summary.extend([f"• {alert}" for alert in critical_alerts[:5]])  # Top 5 alerts
            final_summary.append("")

        if summary_recommendations:
            final_summary.append("📊 WEEKLY WEATHER OVERVIEW")
            final_summary.extend([f"• {rec}" for rec in summary_recommendations])
            final_summary.append("")

        if irrigation_needs:
            final_summary.append("💧 IRRIGATION RECOMMENDATIONS")
            final_summary.extend([f"• {advice}" for advice in irrigation_needs])
            final_summary.append("")

        if general_advice:
            final_summary.append("📋 FARMING ACTIVITY GUIDANCE")
            final_summary.extend([f"• {advice}" for advice in general_advice])

        # Weekly outlook summary
        weekly_outlook = {
            'rain_days': total_rain_days,
            'hot_days': total_hot_days,
            'cold_days': total_cold_days,
            'high_humidity_days': total_high_humidity_days,
            'planting_score': 'Excellent' if len(planting_opportunities) > 3 else 'Good' if len(planting_opportunities) > 1 else 'Limited',
            'harvesting_score': 'Excellent' if len(harvesting_opportunities) > 3 else 'Good' if len(harvesting_opportunities) > 1 else 'Limited',
            'irrigation_priority': 'High' if total_hot_days > 2 or total_rain_days == 0 else 'Low' if total_rain_days > 3 else 'Moderate',
            'disease_risk': 'High' if total_high_humidity_days > 3 else 'Low'
        }

        return {
            'summary': final_summary,
            'weekly_outlook': weekly_outlook,
            'critical_alerts_count': len(critical_alerts),
            'planting_opportunities': len(planting_opportunities),
            'harvesting_opportunities': len(harvesting_opportunities)
        }

    except Exception as e:
        print(f"Weather summary generation error: {e}")
        return {'summary': [], 'weekly_outlook': {}}

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'AGRI HUB - AI Powered Crop & Soil Management'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'AGRI HUB - Smart Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AGRI HUB - Soil Optimizer'

    return render_template('fertilizer.html', title=title)

# render chatbot page

@ app.route('/chatbot')
def chatbot():
    title = 'AGRI HUB - AI Agriculture Assistant'
    return render_template('chatbot.html', title=title)

# render weather recommendations page

@ app.route('/weather-recommendations')
def weather_recommendations():
    title = 'AGRI HUB - Weather-Aware Recommendations'
    return render_template('weather_recommendations.html', title=title)

# render crop comparison dashboard

@ app.route('/crop-comparison')
def crop_comparison():
    title = 'AGRI HUB - Crop Comparison Dashboard'
    return render_template('crop_comparison.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AGRI HUB - Smart Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        weather_data = weather_fetch(city)
        if weather_data != None:
            temperature, humidity = weather_data
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            # If weather API fails, show try again page
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AGRI HUB - Soil Optimizer'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


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
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# chatbot API endpoint

@app.route('/chatbot-api', methods=['POST'])
def chatbot_api():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message.strip():
            return {'response': 'Please enter a message to get started!'}

        # Get response from chatbot
        bot_response = get_chatbot_response(user_message)

        return {'response': bot_response}
    except Exception as e:
        print(f"Chatbot API error: {e}")
        return {'response': 'Sorry, I encountered an error. Please try again.'}

# weather recommendations API endpoint

@app.route('/weather-forecast-api', methods=['POST'])
def weather_forecast_api():
    try:
        data = request.get_json()
        city = data.get('city', '')

        if not city.strip():
            return {'success': False, 'error': 'Please enter a city name'}

        # Get weather forecast
        weather_data = get_weather_forecast(city)

        if weather_data['success']:
            # Generate summary recommendations instead of daily details
            recommendations_summary = generate_weather_summary(weather_data)

            return {
                'success': True,
                'weather': weather_data,
                'recommendations': recommendations_summary
            }
        else:
            return weather_data

    except Exception as e:
        print(f"Weather forecast API error: {e}")
        return {'success': False, 'error': 'Unable to fetch weather data'}

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

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

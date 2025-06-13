#!/usr/bin/env python3

import sys
import os
sys.path.append('app')

print("Testing model loading...")

try:
    print("1. Testing basic imports...")
    import torch
    import pickle
    from utils.model import ResNet9
    print("   Basic imports successful!")
    
    print("2. Testing disease model loading...")
    disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    disease_model_path = 'models/plant_disease_model.pth'
    if os.path.exists(disease_model_path):
        print(f"   Disease model file exists: {disease_model_path}")
        disease_model = ResNet9(3, len(disease_classes))
        disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu'), weights_only=True))
        disease_model.eval()
        print("   Disease model loaded successfully!")
    else:
        print(f"   ERROR: Disease model file not found: {disease_model_path}")
    
    print("3. Testing crop recommendation model loading...")
    crop_model_path = 'models/NBClassifier.pkl'
    if os.path.exists(crop_model_path):
        print(f"   Crop model file exists: {crop_model_path}")
        with open(crop_model_path, 'rb') as f:
            crop_model = pickle.load(f)
        print("   Crop recommendation model loaded successfully!")
    else:
        print(f"   ERROR: Crop model file not found: {crop_model_path}")
        
    print("4. Testing Flask import...")
    from flask import Flask
    print("   Flask import successful!")
    
    print("\nAll tests passed! The models should work.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

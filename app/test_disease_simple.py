#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing disease prediction functionality...")

try:
    # Test imports
    print("1. Testing imports...")
    from utils.disease import disease_dic
    print(f"   ✅ Disease dictionary loaded with {len(disease_dic)} entries")
    
    from utils.model import ResNet9
    print("   ✅ ResNet9 model imported")
    
    import torch
    print("   ✅ PyTorch imported")
    
    from PIL import Image
    print("   ✅ PIL imported")
    
    # Test disease classes
    print("\n2. Testing disease classes...")
    disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___healthy', 'Tomato___healthy']
    print(f"   ✅ Disease classes defined: {len(disease_classes)} classes")
    
    # Test model loading
    print("\n3. Testing model loading...")
    disease_model_path = 'models/plant_disease_model.pth'
    
    if os.path.exists(disease_model_path):
        print(f"   ✅ Model file exists: {disease_model_path}")
        print(f"   📁 File size: {os.path.getsize(disease_model_path)} bytes")
        
        try:
            # Try loading the model
            disease_model = ResNet9(3, len(disease_classes))
            checkpoint = torch.load(disease_model_path, map_location=torch.device('cpu'))
            disease_model.load_state_dict(checkpoint)
            disease_model.eval()
            print("   ✅ Model loaded successfully!")
            
            # Test model with dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = disease_model(dummy_input)
                print(f"   ✅ Model test successful - output shape: {dummy_output.shape}")
                
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            
    else:
        print(f"   ❌ Model file not found: {disease_model_path}")
    
    # Test disease dictionary entries
    print("\n4. Testing disease dictionary...")
    test_classes = ['Apple___healthy', 'Tomato___healthy']
    for cls in test_classes:
        if cls in disease_dic:
            print(f"   ✅ {cls} found in dictionary")
        else:
            print(f"   ❌ {cls} not found in dictionary")
    
    print("\n✅ All tests completed!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\nPress Enter to exit...")
input()

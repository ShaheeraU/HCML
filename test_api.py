#!/usr/bin/env python3
"""
Test script for the Face Recognition API
This script demonstrates how to use the API endpoints
"""

import requests
import base64
import json
import os

# API base URL
BASE_URL = "http://localhost:5002"

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_register_face(name, image_path):
    """Test face registration"""
    print(f"\nTesting face registration for {name}...")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Prepare request data
        data = {
            "name": name,
            "image": image_base64
        }
        
        # Make request
        response = requests.post(f"{BASE_URL}/register", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        return response.status_code == 201
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_recognize_face(image_path, threshold=0.3):
    """Test face recognition"""
    print(f"\nTesting face recognition...")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Prepare request data
        data = {
            "image": image_base64,
            "threshold": threshold
        }
        
        # Make request
        response = requests.post(f"{BASE_URL}/recognize", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_list_users():
    """Test listing all users"""
    print(f"\nTesting list users...")
    try:
        response = requests.get(f"{BASE_URL}/users")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("Face Recognition API Test Script")
    print("=" * 40)
    
    # Test health check first
    if not test_health_check():
        print("Health check failed. Make sure the API is running.")
        return
    
    # Test with sample images from data folder
    data_dir = "data"
    if os.path.exists(data_dir):
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            print(f"\nFound {len(image_files)} image files in data directory")
            
            # Test registration with first image
            first_image = os.path.join(data_dir, image_files[0])
            # Generate unique name with timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            name = f"TestUser_{timestamp}_{os.path.splitext(image_files[0])[0]}"
            
            if test_register_face(name, first_image):
                print("Registration successful!")
                
                # Test recognition with the same image
                if test_recognize_face(first_image):
                    print("Recognition successful!")
                
                # Test listing users
                test_list_users()
            else:
                print("Registration failed!")
        else:
            print("No image files found in data directory")
    else:
        print("Data directory not found")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()

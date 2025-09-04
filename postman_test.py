#!/usr/bin/env python3
"""
Simple test script to demonstrate proper API usage
This shows the exact format needed for Postman
"""

import requests
import base64
import json

# API base URL
BASE_URL = "http://localhost:5002"

def test_register():
    """Test registration endpoint"""
    print("=== Testing Registration Endpoint ===")
    
    # Create a simple test image (1x1 pixel)
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    # Request data
    data = {
        "name": "PostmanTestUser",
        "image": test_image
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"URL: {BASE_URL}/register")
    print(f"Headers: {headers}")
    print(f"Body: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/register", json=data, headers=headers)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 201
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_recognize():
    """Test recognition endpoint"""
    print("\n=== Testing Recognition Endpoint ===")
    
    # Use the same test image
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    # Request data
    data = {
        "image": test_image,
        "threshold": 0.3
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"URL: {BASE_URL}/recognize")
    print(f"Headers: {headers}")
    print(f"Body: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/recognize", json=data, headers=headers)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    
    print(f"URL: {BASE_URL}/health")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    print("Postman Test Script for Face Recognition API")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("Health check failed. Make sure the API is running on port 5002")
        return
    
    # Test registration
    test_register()
    
    # Test recognition
    test_recognize()
    
    print("\n" + "=" * 50)
    print("Test completed! Use the above format in Postman.")

if __name__ == "__main__":
    main()

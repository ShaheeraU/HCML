from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import json
from datetime import datetime
from face_recognition_system import FaceRecognitionSystem
import base64
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize face recognition system
fr_system = FaceRecognitionSystem()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Database file for storing user records
DATABASE_FILE = 'user_database.json'

def load_user_database():
    """Load user database from JSON file"""
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_database(database):
    """Save user database to JSON file"""
    with open(DATABASE_FILE, 'w') as f:
        json.dump(database, f, indent=2)

def save_image_from_base64(base64_string, filename):
    """Save base64 encoded image to file"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 and save
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        filepath = os.path.join('data', filename)
        image.save(filepath, 'JPEG', quality=95)
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'face_recognition_system': 'initialized'
    })

@app.route('/register', methods=['POST'])
def register_face():
    """Register a new face with name and image"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        name = data.get('name')
        image_base64 = data.get('image')
        
        if not name or not image_base64:
            return jsonify({
                'error': 'Both name and image are required'
            }), 400
        
        # Generate unique ID
        user_id = str(uuid.uuid4())
        
        # Save image to file
        filename = f"{user_id}.jpg"
        image_path = save_image_from_base64(image_base64, filename)
        
        print(f"Processing registration for user: {name}")
        print(f"Generated UUID: {user_id}")
        print(f"Image saved to: {image_path}")
        
        if not image_path:
            return jsonify({'error': 'Failed to save image'}), 500
        
        # Register face in the recognition system
        try:
            registration_result = fr_system.register_user(name, image_path)
            print(f"Registration result: {registration_result}")
            
            if not registration_result:
                return jsonify({'error': 'Face registration failed - no result returned'}), 500
        except Exception as e:
            print(f"Registration error: {e}")
            return jsonify({'error': f'Face registration failed: {str(e)}'}), 500
        
        # Store additional user metadata
        user_database = load_user_database()
        user_database[user_id] = {
            'name': name,
            'image_path': image_path,
            'registered_at': datetime.now().isoformat(),
            'face_recognition_id': registration_result
        }
        save_user_database(user_database)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'name': name,
            'message': 'Face registered successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Recognize a face from the database"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        image_base64 = data.get('image')
        threshold = data.get('threshold', 0.3)
        
        if not image_base64:
            return jsonify({'error': 'Image is required'}), 400
        
        # Save temporary image for recognition
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        image_path = save_image_from_base64(image_base64, temp_filename)
        
        if not image_path:
            return jsonify({'error': 'Failed to save image'}), 500
        
        try:
            # Perform face recognition
            user_id, name, similarity = fr_system.match_face(image_path, threshold)
            
            # Clean up temporary file
            if os.path.exists(image_path):
                os.remove(image_path)
            
            if user_id and name:
                # Get additional user data
                user_database = load_user_database()
                user_data = user_database.get(user_id, {})
                
                return jsonify({
                    'success': True,
                    'match_found': True,
                    'user_id': user_id,
                    'name': name,
                    'similarity': float(similarity),
                    'threshold': float(threshold),
                    'registered_at': user_data.get('registered_at'),
                    'image_path': user_data.get('image_path')
                })
            else:
                return jsonify({
                    'success': True,
                    'match_found': False,
                    'message': 'No matching face found',
                    'best_similarity': float(similarity),
                    'threshold': float(threshold)
                })
                
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(image_path):
                os.remove(image_path)
            raise e
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users', methods=['GET'])
def list_users():
    """List all registered users"""
    try:
        user_database = load_user_database()
        
        users = []
        for user_id, data in user_database.items():
            users.append({
                'user_id': user_id,
                'name': data['name'],
                'registered_at': data['registered_at'],
                'image_path': data['image_path']
            })
        
        return jsonify({
            'success': True,
            'total_users': len(users),
            'users': users
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user details"""
    try:
        user_database = load_user_database()
        
        if user_id not in user_database:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_database[user_id]
        
        return jsonify({
            'success': True,
            'user': {
                'user_id': user_id,
                'name': user_data['name'],
                'registered_at': user_data['registered_at'],
                'image_path': user_data['image_path']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user and their face data"""
    try:
        user_database = load_user_database()
        
        if user_id not in user_database:
            return jsonify({'error': 'User not found'}), 404
        
        user_data = user_database[user_id]
        
        # Remove image file
        if os.path.exists(user_data['image_path']):
            os.remove(user_data['image_path'])
        
        # Remove from user database
        del user_database[user_id]
        save_user_database(user_database)
        
        # Note: Face recognition system database is separate and would need additional cleanup
        
        return jsonify({
            'success': True,
            'message': f'User {user_data["name"]} deleted successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Test the face recognition system first
    if fr_system.test_model():
        print("Face recognition system initialized successfully!")
        app.run(debug=True, host='0.0.0.0', port=5002)
    else:
        print("Face recognition system initialization failed!")
        exit(1)

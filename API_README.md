# Face Recognition API

A Flask-based REST API for face registration and recognition using the MobileFaceNet model.

## Features

- **Face Registration**: Register new faces with names and unique IDs
- **Face Recognition**: Recognize faces from the database
- **User Management**: List, view, and delete registered users
- **File-based Storage**: Images stored in `data/` folder, metadata in JSON
- **Base64 Image Support**: Accept images as base64 encoded strings

## API Endpoints

### 1. Health Check
```
GET /health
```
Returns the health status of the API and face recognition system.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "face_recognition_system": "initialized"
}
```

### 2. Register Face
```
POST /register
```
Register a new face with name and image.

**Request Body:**
```json
{
  "name": "John Doe",
  "image": "base64_encoded_image_string"
}
```

**Response (Success - 201):**
```json
{
  "success": true,
  "user_id": "uuid-string",
  "name": "John Doe",
  "message": "Face registered successfully"
}
```

**Response (Error - 400/500):**
```json
{
  "error": "Error message"
}
```

### 3. Recognize Face
```
POST /recognize
```
Recognize a face from the database.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "threshold": 0.3
}
```

**Response (Match Found):**
```json
{
  "success": true,
  "match_found": true,
  "user_id": "uuid-string",
  "name": "John Doe",
  "similarity": 0.85,
  "threshold": 0.3,
  "registered_at": "2024-01-15T10:30:00",
  "image_path": "data/uuid-string.jpg"
}
```

**Response (No Match):**
```json
{
  "success": true,
  "match_found": false,
  "message": "No matching face found",
  "best_similarity": 0.25,
  "threshold": 0.3
}
```

### 4. List Users
```
GET /users
```
List all registered users.

**Response:**
```json
{
  "success": true,
  "total_users": 2,
  "users": [
    {
      "user_id": "uuid-1",
      "name": "John Doe",
      "registered_at": "2024-01-15T10:30:00",
      "image_path": "data/uuid-1.jpg"
    },
    {
      "user_id": "uuid-2",
      "name": "Jane Smith",
      "registered_at": "2024-01-15T11:00:00",
      "image_path": "data/uuid-2.jpg"
    }
  ]
}
```

### 5. Get User Details
```
GET /users/<user_id>
```
Get details of a specific user.

**Response:**
```json
{
  "success": true,
  "user": {
    "user_id": "uuid-string",
    "name": "John Doe",
    "registered_at": "2024-01-15T10:30:00",
    "image_path": "data/uuid-string.jpg"
  }
}
```

### 6. Delete User
```
DELETE /users/<user_id>
```
Delete a user and their associated data.

**Response:**
```json
{
  "success": true,
  "message": "User John Doe deleted successfully"
}
```

## Installation & Setup

### 1. Install Dependencies
```bash
# Install main requirements
pip install -r requirements.txt

# Install API-specific requirements
pip install -r api_requirements.txt
```

### 2. Run the API
```bash
python api.py
```

The API will start on `http://localhost:5000`

### 3. Test the API
```bash
python test_api.py
```

## Usage Examples

### Python Client Example

```python
import requests
import base64

# Encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Register a face
def register_face(name, image_path):
    url = "http://localhost:5000/register"
    data = {
        "name": name,
        "image": encode_image(image_path)
    }
    response = requests.post(url, json=data)
    return response.json()

# Recognize a face
def recognize_face(image_path, threshold=0.3):
    url = "http://localhost:5000/recognize"
    data = {
        "image": encode_image(image_path),
        "threshold": threshold
    }
    response = requests.post(url, json=data)
    return response.json()

# Usage
result = register_face("John Doe", "path/to/image.jpg")
print(f"Registered with ID: {result['user_id']}")

match = recognize_face("path/to/query_image.jpg")
if match['match_found']:
    print(f"Recognized: {match['name']}")
else:
    print("No match found")
```

### cURL Examples

**Register a face:**
```bash
curl -X POST http://localhost:5000/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "image": "base64_encoded_image_string"
  }'
```

**Recognize a face:**
```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_string",
    "threshold": 0.3
  }'
```

**List all users:**
```bash
curl http://localhost:5000/users
```

## File Structure

```
mobile_face_recognition/
├── api.py                 # Main Flask API
├── api_requirements.txt   # API dependencies
├── test_api.py           # Test script
├── face_recognition_system.py  # Core face recognition logic
├── data/                 # Image storage folder
├── user_database.json    # User metadata database
└── face_database.json    # Face recognition embeddings
```

## Configuration

- **Max file size**: 16MB (configurable in `api.py`)
- **Image format**: JPEG (converted automatically)
- **Threshold**: Default 0.3 for face matching (configurable per request)
- **Port**: 5000 (configurable in `api.py`)

## Error Handling

The API includes comprehensive error handling:
- Input validation
- File processing errors
- Face recognition failures
- Database errors
- Proper HTTP status codes

## Security Notes

- Images are stored locally in the `data/` folder
- No authentication implemented (add as needed for production)
- Consider rate limiting for production use
- Validate image formats and sizes

## Troubleshooting

### Common Issues

1. **Face recognition system fails to initialize**
   - Check that all dependencies are installed
   - Verify model files exist in the correct locations

2. **Image processing errors**
   - Ensure images are valid JPEG/PNG files
   - Check image size (max 16MB)

3. **No faces detected**
   - Ensure images contain clear, front-facing faces
   - Check image quality and lighting

### Debug Mode

The API runs in debug mode by default. Check the console output for detailed error messages.

## Production Deployment

For production use, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Adding authentication and authorization
- Implementing rate limiting
- Using a proper database (PostgreSQL, MongoDB)
- Adding logging and monitoring
- Using HTTPS
- Implementing image compression and optimization

# Mobile Face Recognition System

A comprehensive face recognition system built with PyTorch and Expo React Native, featuring MobileFaceNet optimized for mobile deployment using AdaDistill knowledge distillation.

## Features

- **Advanced Face Detection**: MTCNN-based face detection and alignment
- **Mobile-Optimized Model**: MobileFaceNet with AdaDistill optimization
- **Cross-Platform Support**: Expo React Native app for iOS and Android
- **RESTful API**: Flask-based backend with comprehensive endpoints
- **Docker Support**: Containerized deployment with persistent data storage
- **Real-time Processing**: Camera and gallery image processing
- **User Management**: Complete registration and recognition system

## Project Structure

```
mobile_face_recognition/
├── api.py                          # Flask REST API server
├── app.py                          # Standalone Python application
├── face_recognition_system.py      # Core face recognition system
├── mobile_facenet_converter.py     # Model conversion utilities
├── face_database.json             # User face database
├── user_database.json             # User metadata database
├── requirements.txt                # Python dependencies
├── api_requirements.txt            # API-specific dependencies
├── Dockerfile                      # Docker configuration
├── model/                          # Model files
│   ├── MFN_AdaArcDistill_backbone.pth
│   ├── mobile_config.json
│   └── mobilefacenet_mobile.onnx
├── data/                           # User uploaded images
│   ├── *.jpg                       # User face images
│   └── *.jpeg                      # Sample images
├── output/                         # Processed images
└── my-app/                         # Expo React Native application
    ├── app/                        # App screens and navigation
    ├── components/                 # Reusable components
    ├── constants/                  # App constants and configuration
    └── package.json                # Node.js dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- Node.js and npm (for Expo app)
- Expo CLI

## Docker Deployment (Recommended)

1. **Build the Docker image**
   ```bash
   docker build -t face-recognition-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 5002:5002 -v face-recognition-data:/app/data face-recognition-app
   ```

The API will be available at `http://localhost:5002`

## Local Development Setup

### Backend API

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r api_requirements.txt
   ```

2. **Run the Flask API server**
   ```bash
   python api.py
   ```

### Standalone Python Script

1. **Run the standalone application**
   ```bash
   python app.py
   ```

### How to use

- You can edit the name and image path for the user registration in the app.py
- You can edit the path of the image to match the face in app.py

### Example Usage

```python
from face_recognition_system import FaceRecognitionSystem

# Initialize system
fr_system = FaceRecognitionSystem()

# Register users with custom names and images
fr_system.register_user("John Doe", "data/john.jpg")
fr_system.register_user("Jane Smith", "data/jane.jpg")

# Match a face with custom threshold
user_id, name, similarity = fr_system.match_face(
    "data/test_image.jpg",
    threshold=0.3  # Adjust threshold: 0.2 (lenient) to 0.5 (strict)
)

if user_id:
    print(f"Recognized: {name} (Confidence: {similarity:.3f})")
else:
    print("No match found")
```

## Expo React Native App

1. **Install dependencies**
   ```bash
   cd my-app
   npm install
   ```

2. **Start the Expo development server**
   ```bash
   npx expo start
   ```

3. **Run on device/simulator**
   - Scan QR code with Expo Go app (iOS/Android)
   - Press `i` for iOS simulator
   - Press `a` for Android emulator

### Expo App Features

- **User Registration**: Register new users with face photos
- **Face Recognition**: Real-time face matching against registered users
- **User Management**: View and manage registered users
- **Camera Integration**: Capture photos directly from the app
- **Gallery Support**: Process images from device gallery

## Model Files

Ensure these files are in the `model/` directory:

- `MFN_AdaArcDistill_backbone.pth` - Trained MobileFaceNet model
- `mobile_config.json` - Model configuration
- `mobilefacenet_mobile.onnx` - Converted ONNX model for mobile

## Flask API Configuration

### API Endpoints

The Flask API provides the following endpoints:

- `GET /health` - Health check endpoint
- `POST /register` - Register a new user with face image
- `POST /recognize` - Recognize a face from uploaded image
- `GET /users` - List all registered users
- `GET /users/<user_id>` - Get specific user details
- `DELETE /users/<user_id>` - Delete a user

### API Usage Examples

**Register a new user:**
```bash
curl -X POST http://localhost:5002/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "image": "base64_encoded_image_data"
  }'
```

**Recognize a face:**
```bash
curl -X POST http://localhost:5002/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "threshold": 0.3
  }'
```

**List all users:**
```bash
curl http://localhost:5002/users
```

### Recognition Threshold

Adjust the similarity threshold in API requests:

- **0.2**: More lenient matching (may increase false positives)
- **0.3**: Balanced matching (recommended)
- **0.5**: Stricter matching (may increase false negatives)

### Environment Variables

Configure the Flask app with these environment variables:

```bash
export FLASK_APP=api.py
export FLASK_ENV=development  # or production
export PYTHONPATH=/app
```

## Requirements

### Python Dependencies

See `requirements.txt` and `api_requirements.txt` for complete dependencies:

- torch
- torchvision
- facenet-pytorch
- flask
- werkzeug
- Pillow
- numpy

### Node.js Dependencies

See `my-app/package.json` for Expo app dependencies:

- expo
- react-native
- @expo/vector-icons

## Docker Volume Management

The Docker setup uses a named volume for persistent data storage:

```bash
# View volume details
docker volume inspect face-recognition-data

# Backup data (optional)
docker run --rm -v face-recognition-data:/data -v $(pwd):/backup alpine tar czf /backup/face-data-backup.tar.gz -C /data .

# Restore data (optional)
docker run --rm -v face-recognition-data:/data -v $(pwd):/backup alpine tar xzf /backup/face-data-backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure model files are in the `model/` directory
2. **Face not detected**: Use well-lit images with clear frontal faces
3. **Low recognition accuracy**: Adjust threshold or use higher quality images
4. **Docker build fails**: Check that all dependencies are compatible with Python 3.10
5. **API connection issues**: Ensure the Flask server is running on port 5002
6. **Expo app issues**: Run `npx expo start --clear` to clear cache

### Performance Tips

- Use high-quality, well-lit images for registration
- Ensure faces are clearly visible and frontal
- Test with different lighting conditions
- Adjust threshold based on your accuracy requirements
- Use Docker volumes for persistent data storage
- Monitor container logs for debugging: `docker logs <container_id>`

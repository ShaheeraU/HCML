<<<<<<< HEAD
READ ME
=======
# Mobile Face Recognition System

A comprehensive face recognition system built with PyTorch and Flutter, featuring MobileFaceNet optimized for mobile deployment using AdaDistill knowledge distillation.

## Features

- **Advanced Face Detection**: MTCNN-based face detection and alignment
- **Mobile-Optimized Model**: MobileFaceNet with AdaDistill optimization
- **Cross-Platform Support**: Flutter app for iOS and Android
- **ONNX Runtime Integration**: Optimized inference on mobile devices
- **Real-time Processing**: Camera and gallery image processing
- **User Management**: Complete registration and recognition system

## Project Structure

```
mobile_face_recognition/
├── app.py                          # Main Python application
├── face_recognition_system.py      # Core face recognition system
├── mobile_facenet_converter.py     # Model conversion utilities
├── face_database.json             # User face database
├── requirements.txt                # Python dependencies
├── model/                          # Model files
│   ├── MFN_AdaArcDistill_backbone.pth
│   ├── mobile_config.json
│   └── mobilefacenet_mobile.onnx
├── data/                           # Sample images
│   ├── chris_hemsworth_sample1.jpg
│   ├── chris_hemsworth_sample2.jpg
│   └── prit_sample1.jpeg
├── output/                         # Processed images
└── flutter_app/                    # Flutter mobile application
    └── flutter_application_1/
```

## Setup

### Prerequisites

- Python 3.8+
- Flutter SDK
- PyTorch
- OpenCV

## Python Standalone Script Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
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

## Flutter Mobile App

1. **Install Flutter dependencies**

   ```bash
   cd flutter_app/flutter_application_1
   flutter pub get
   ```

2. **Run the app**
   ```bash
   flutter run
   ```

### Flutter App Features

- **Face Detection Test**: Upload images and test face detection
- **ONNX Integration**: Test model inference with face embeddings
- **Gallery Support**: Process images from device gallery
- **Real-time Analysis**: Face positioning and quality feedback

## Model Files

Ensure these files are in the `model/` directory:

- `MFN_AdaArcDistill_backbone.pth` - Trained MobileFaceNet model
- `mobile_config.json` - Model configuration
- `mobilefacenet_mobile.onnx` - Converted ONNX model for mobile

## Configuration

### Recognition Threshold

Adjust the similarity threshold in `app.py`:

- **0.2**: More lenient matching (may increase false positives)
- **0.3**: Balanced matching (recommended)
- **0.5**: Stricter matching (may increase false negatives)

### Custom User Data

Edit the registration section in `app.py`:

```python
# Register your own users
fr_system.register_user("Your Name", "path/to/your/photo.jpg")
fr_system.register_user("Friend Name", "path/to/friend/photo.jpg")

# Test with your own images
user_id, name, similarity = fr_system.match_face("path/to/test/image.jpg")
```

## Requirements

See `requirements.txt` for complete Python dependencies:

- torch
- torchvision
- facenet-pytorch
- opencv-python
- Pillow
- numpy

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure model files are in the `model/` directory
2. **Face not detected**: Use well-lit images with clear frontal faces
3. **Low recognition accuracy**: Adjust threshold or use higher quality images
4. **Flutter build issues**: Run `flutter clean` and `flutter pub get`

### Performance Tips

- Use high-quality, well-lit images for registration
- Ensure faces are clearly visible and frontal
- Test with different lighting conditions
- Adjust threshold based on your accuracy requirements
>>>>>>> b10c54c (Initial commit)

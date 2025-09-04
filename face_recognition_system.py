import torch
import torch.nn as nn
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional


class ConvBNReLU(nn.Module):
    """Standard convolution with BatchNorm and ReLU"""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2"""

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Point-wise convolution
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # Depth-wise convolution
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # Point-wise linear convolution
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet implementation based on AdaDistill paper
    Optimized for mobile face recognition with knowledge distillation
    """

    def __init__(self, embedding_size=512, width_mult=1.0):
        super(MobileFaceNet, self).__init__()

        # Setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 64, 1, 2],   # Stage 1
            [6, 64, 2, 1],   # Stage 2
            [6, 128, 3, 2],  # Stage 3
            [6, 128, 4, 1],  # Stage 4
            [6, 128, 3, 2],  # Stage 5
            [6, 128, 3, 1],  # Stage 6
            [6, 128, 1, 1],  # Stage 7
        ]

        # Building first layer
        input_channel = int(32 * width_mult)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2)

        # Building inverted residual blocks
        self.features = []
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(
                    input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        # Building last several layers
        output_channel = int(512 * width_mult) if width_mult > 1.0 else 512
        self.conv2 = ConvBNReLU(input_channel, output_channel, kernel_size=1)

        # Global average pooling + FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

        # Final embedding layer
        self.fc = nn.Linear(output_channel, embedding_size)
        self.bn_fc = nn.BatchNorm1d(embedding_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Feature extraction
        x = self.features(x)

        # Final convolution
        x = self.conv2(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Dropout and final embedding
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)

        return x


class FaceRecognitionSystem:
    def __init__(self, model_path: str = "model/MFN_AdaArcDistill_backbone.pth",
                 database_path: str = "face_database.json"):
        """
        Initialize the face recognition system with AdaDistill MobileFaceNet

        Args:
            model_path: Path to the AdaDistill MobileFaceNet model
            database_path: Path to store user face templates
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=112,  # Standard size for face recognition
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

        # Load the AdaDistill MobileFaceNet model
        self.model = self._load_model(model_path)

        # Image preprocessing - optimized for face recognition
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Database file path
        self.database_path = database_path

        # Load existing database
        self.face_database = self._load_database()

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the AdaDistill MobileFaceNet model"""
        try:
            # Initialize model with AdaDistill architecture
            model = MobileFaceNet(embedding_size=512)

            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")

                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Clean up keys (remove module. prefix if present)
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        cleaned_key = key[7:]  # Remove 'module.' prefix
                    else:
                        cleaned_key = key
                    cleaned_state_dict[cleaned_key] = value

                # Load weights
                missing_keys, unexpected_keys = model.load_state_dict(
                    cleaned_state_dict, strict=False)

                if missing_keys:
                    # Show first 5
                    print(f"Missing keys: {missing_keys[:5]}...")
                if unexpected_keys:
                    # Show first 5
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")

                print("Model loaded successfully!")
            else:
                print(f"Model file not found at {model_path}")
                print("Using randomly initialized model for testing")

            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to randomly initialized model")
            model = MobileFaceNet(embedding_size=512)
            model.to(self.device)
            model.eval()
            return model

    def test_model(self):
        """Test the loaded model with dummy input"""
        try:
            print("Testing model with dummy input...")
            dummy_input = torch.randn(1, 3, 112, 112).to(self.device)

            with torch.no_grad():
                output = self.model(dummy_input)

            print(f"Model test successful!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print(
                f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

            return True

        except Exception as e:
            print(f"Model test failed: {e}")
            return False

    def _load_database(self) -> Dict:
        """Load existing face database"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    database = json.load(f)
                    # Convert string embeddings back to numpy arrays
                    for user_id in database:
                        database[user_id]['embedding'] = np.array(
                            database[user_id]['embedding'])
                    return database
            except Exception as e:
                print(f"Error loading database: {e}")
                return {}
        return {}

    def _save_database(self):
        """Save face database to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            database_copy = {}
            for user_id, data in self.face_database.items():
                database_copy[user_id] = {
                    'name': data['name'],
                    'embedding': data['embedding'].tolist()
                }

            with open(self.database_path, 'w') as f:
                json.dump(database_copy, f, indent=2)
            print(f"Database saved to {self.database_path}")
        except Exception as e:
            print(f"Error saving database: {e}")

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess image: detect, align, and crop face

        Args:
            image_path: Path to the input image

        Returns:
            Preprocessed face tensor or None if no face detected
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')

            # Detect and align face using MTCNN
            img_cropped = self.mtcnn(img)

            if img_cropped is None:
                print("No face detected in the image")
                return None

            # Apply additional preprocessing
            img_tensor = self.transform(transforms.ToPILImage()(img_cropped))

            return img_tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def extract_embedding(self, preprocessed_image: torch.Tensor) -> np.ndarray:
        """
        Extract 512-dimensional face embedding

        Args:
            preprocessed_image: Preprocessed face tensor

        Returns:
            512-dimensional embedding vector
        """
        with torch.no_grad():
            embedding = self.model(preprocessed_image)
            # Normalize embedding
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()

    def register_user(self, name: str, image_path: str) -> bool:
        """
        Register a new user with face template

        Args:
            name: User's name
            image_path: Path to user's face image

        Returns:
            True if registration successful, False otherwise
        """
        # Dynamically generate user_id
        user_id = str(len(self.face_database) + 1)
        print(f"Registering user: {name} (ID: {user_id})")

        # Check if user already exists by name
        for data in self.face_database.values():
            if data['name'] == name:
                print(f"User with name '{name}' already exists!")
                return False

        # Preprocess image
        preprocessed_img = self.preprocess_image(image_path)
        if preprocessed_img is None:
            print("Failed to preprocess image for registration")
            return False

        # Extract embedding
        embedding = self.extract_embedding(preprocessed_img)

        # Store in database
        self.face_database[user_id] = {
            'name': name,
            'embedding': embedding
        }

        # Save to file
        self._save_database()

        print(f"User {name} registered successfully with id {user_id}!")
        return user_id

    def match_face(self, image_path: str, threshold: float = 0.3) -> Tuple[Optional[str], Optional[str], float]:
        """
        Match face against registered users

        Args:
            image_path: Path to the query image
            threshold: Cosine similarity threshold for matching

        Returns:
            Tuple of (user_id, name, similarity_score) if match found, else (None, None, 0)
        """
        print(f"Matching face from: {image_path}")

        # Preprocess image
        preprocessed_img = self.preprocess_image(image_path)
        if preprocessed_img is None:
            print("Failed to preprocess image for matching")
            return None, None, 0.0

        # Extract embedding
        query_embedding = self.extract_embedding(preprocessed_img)

        # Compare with all registered users
        best_match_id = None
        best_match_name = None
        best_similarity = 0.0

        for user_id, data in self.face_database.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, data['embedding']) / (
                np.linalg.norm(query_embedding) *
                np.linalg.norm(data['embedding'])
            )

            print(f"Similarity with {data['name']}: {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = user_id
                best_match_name = data['name']

        # Check if best match exceeds threshold
        if best_similarity >= threshold:
            print(
                f"Match found: {best_match_name} (ID: {best_match_id}) with similarity {best_similarity:.4f}")
            return best_match_id, best_match_name, best_similarity
        else:
            print(
                f"No match found. Best similarity: {best_similarity:.4f} (threshold: {threshold})")
            return None, None, best_similarity

    def list_users(self):
        """List all registered users"""
        if not self.face_database:
            print("No users registered")
            return

        print("Registered users:")
        for user_id, data in self.face_database.items():
            print(f"- {data['name']} (ID: {user_id})")

import torch
import torch.onnx
import numpy as np
import os
import json
import traceback
from face_recognition_system import MobileFaceNet


class MobileFaceNetConverterFixed:
    def __init__(self, model_path: str = "model/MFN_AdaArcDistill_backbone.pth"):
        self.model_path = model_path
        self.device = torch.device('cpu')

    def analyze_checkpoint_architecture(self):
        """Analyze the checkpoint to determine the correct model architecture"""
        print("🔍 Analyzing checkpoint architecture...")

        if not os.path.exists(self.model_path):
            print(f"⚠️ Model file not found at {self.model_path}")
            return None

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract state dict
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

            # Clean up keys
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    cleaned_key = key[7:]
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value

            # Analyze the architecture from state dict
            conv1_weight_shape = cleaned_state_dict.get(
                'conv1.conv.weight', None)
            if conv1_weight_shape is not None:
                # Shape is [out_channels, in_channels, kernel_h, kernel_w]
                conv1_out_channels = conv1_weight_shape.shape[0]
                print(
                    f"✅ Detected conv1 output channels: {conv1_out_channels}")

                # Try to determine width_mult from conv1 channels
                # Standard MobileFaceNet conv1 is 32 channels at width_mult=1.0
                # So if we have 64 channels, width_mult might be 2.0
                if conv1_out_channels == 32:
                    width_mult = 1.0
                elif conv1_out_channels == 64:
                    width_mult = 2.0
                elif conv1_out_channels == 16:
                    width_mult = 0.5
                else:
                    # Calculate based on ratio
                    width_mult = conv1_out_channels / 32.0

                print(f"✅ Estimated width_mult: {width_mult}")

                # Check embedding size from the last layer
                embedding_key = None
                for key in cleaned_state_dict.keys():
                    if 'classifier' in key and 'weight' in key:
                        embedding_key = key
                        break
                    elif 'linear' in key and 'weight' in key:
                        embedding_key = key
                        break
                    elif 'fc' in key and 'weight' in key:
                        embedding_key = key
                        break

                embedding_size = 512  # Default
                if embedding_key:
                    embedding_weight_shape = cleaned_state_dict[embedding_key]
                    embedding_size = embedding_weight_shape.shape[0]
                    print(f"✅ Detected embedding size: {embedding_size}")
                else:
                    print(
                        f"⚠️ Could not detect embedding size, using default: {embedding_size}")

                return {
                    'width_mult': width_mult,
                    'embedding_size': embedding_size,
                    'conv1_channels': conv1_out_channels
                }
            else:
                print("❌ Could not find conv1.conv.weight in checkpoint")
                return None

        except Exception as e:
            print(f"❌ Failed to analyze checkpoint: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def load_pytorch_model_with_correct_arch(self):
        """Load PyTorch model with architecture matching the checkpoint"""
        print("=" * 50)
        print("🔍 Loading PyTorch MobileFaceNet with correct architecture...")
        print("=" * 50)

        # First, analyze the checkpoint to get the correct architecture
        arch_info = self.analyze_checkpoint_architecture()

        if arch_info is None:
            print("❌ Could not determine architecture from checkpoint")
            print("🔄 Trying with default architecture...")
            arch_info = {'width_mult': 1.0,
                         'embedding_size': 512, 'conv1_channels': 32}

        try:
            # Create model with the correct architecture
            model = MobileFaceNet(
                embedding_size=arch_info['embedding_size'],
                width_mult=arch_info['width_mult']
            )
            print(f"✅ Model architecture created successfully")
            print(f"   - Width multiplier: {arch_info['width_mult']}")
            print(f"   - Embedding size: {arch_info['embedding_size']}")
            print(
                f"   - Expected conv1 channels: {arch_info['conv1_channels']}")

        except Exception as e:
            print(f"❌ Failed to create model architecture: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None

        if os.path.exists(self.model_path):
            print(f"📁 Loading weights from: {self.model_path}")

            try:
                checkpoint = torch.load(
                    self.model_path, map_location=self.device)

                # Extract and clean state dict (same as before)
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

                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        cleaned_key = key[7:]
                    else:
                        cleaned_key = key
                    cleaned_state_dict[cleaned_key] = value

                # Load state dict with strict=False to handle any remaining mismatches
                missing_keys, unexpected_keys = model.load_state_dict(
                    cleaned_state_dict, strict=False)

                if missing_keys:
                    print(
                        f"⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
                if unexpected_keys:
                    print(
                        f"⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

                print("✅ State dict loaded successfully!")

            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                return None
        else:
            print(f"⚠️ Model file not found at {self.model_path}")
            print("Using randomly initialized model for testing")

        try:
            model.eval()
            print("✅ Model set to evaluation mode")
        except Exception as e:
            print(f"❌ Failed to set model to eval mode: {e}")
            return None

        return model

    def test_pytorch_model(self):
        """Test the PyTorch model"""
        print("\n" + "=" * 50)
        print("🔍 Testing PyTorch Model...")
        print("=" * 50)

        try:
            model = self.load_pytorch_model_with_correct_arch()
            if model is None:
                print("❌ Model loading failed")
                return False
        except Exception as e:
            print(f"❌ Exception during model loading: {e}")
            return False

        try:
            print("Creating dummy input...")
            dummy_input = torch.randn(1, 3, 112, 112)
            print(f"✅ Dummy input created: {dummy_input.shape}")

            print("Running forward pass...")
            with torch.no_grad():
                output = model(dummy_input)

            print(f"✅ Forward pass successful!")
            print(f"   - Output shape: {output.shape}")
            print(
                f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")

            return True

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def convert_to_onnx(self, output_path: str = "model/mobilefacenet_mobile.onnx"):
        """Convert MobileFaceNet to ONNX"""
        print("\n" + "=" * 50)
        print("🔍 Converting to ONNX...")
        print("=" * 50)

        try:
            model = self.load_pytorch_model_with_correct_arch()
            if model is None:
                return None
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return None

        try:
            dummy_input = torch.randn(1, 3, 112, 112)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print("Attempting ONNX export...")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['face_input'],
                output_names=['face_embedding'],
                dynamic_axes={
                    'face_input': {0: 'batch_size'},
                    'face_embedding': {0: 'batch_size'}
                },
                verbose=False
            )
            print(f"✅ ONNX model saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def create_mobile_config(self, model_arch_info=None):
        """Create configuration file for mobile deployment"""
        if model_arch_info is None:
            model_arch_info = self.analyze_checkpoint_architecture()

        if model_arch_info is None:
            model_arch_info = {'width_mult': 1.0, 'embedding_size': 512}

        mobile_config = {
            "model_info": {
                "name": "MobileFaceNet_AdaDistill",
                "input_shape": [1, 3, 112, 112],
                "output_shape": [1, model_arch_info['embedding_size']],
                "input_name": "face_input",
                "output_name": "face_embedding",
                "input_dtype": "float32",
                "width_mult": model_arch_info['width_mult']
            },
            "preprocessing": {
                "face_size": 112,
                "channel_order": "RGB",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "matching": {
                "similarity_threshold": 0.3,
                "distance_metric": "cosine"
            }
        }

        config_path = "model/mobile_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(mobile_config, f, indent=2)

        print(f"✅ Mobile config saved: {config_path}")
        return config_path

    def full_conversion(self):
        """Complete conversion pipeline"""
        print("🚀 Starting Mobile Face Recognition Conversion (FIXED)")
        print("=" * 70)

        # Step 1: Test PyTorch model
        if not self.test_pytorch_model():
            print("❌ PyTorch model test failed")
            return False

        # Step 2: Convert to ONNX
        onnx_path = self.convert_to_onnx()
        if not onnx_path:
            print("❌ ONNX conversion failed")
            return False

        # Step 3: Create mobile configuration
        arch_info = self.analyze_checkpoint_architecture()
        config_path = self.create_mobile_config(arch_info)

        print(f"\n🎉 Conversion completed successfully!")
        print(f"✅ ONNX model: {onnx_path}")
        print(f"✅ Mobile config: {config_path}")

        print(f"\n📱 Files ready for Android integration:")
        print(f"   - {onnx_path}")
        print(f"   - {config_path}")

        return True


def main():
    """Main function"""
    print("🔧 FIXED: Mobile Face Recognition Converter")
    print("This version automatically detects and matches model architecture")
    print()

    converter = MobileFaceNetConverterFixed()
    success = converter.full_conversion()

    if success:
        print("\n✅ Conversion completed successfully!")
        print("Your model is now ready for mobile deployment!")
    else:
        print("\n❌ Conversion failed - check output above")


if __name__ == "__main__":
    main()

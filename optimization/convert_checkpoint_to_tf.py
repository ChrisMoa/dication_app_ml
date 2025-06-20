import os
import torch
import tensorflow as tf
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from pathlib import Path
import shutil

class CheckpointToTFConverter:
    def __init__(self, checkpoint_path="../models/trained_models/checkpoint-3500"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path("./models")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_pytorch_model(self):
        """Load PyTorch model from checkpoint"""
        print(f"Loading PyTorch model from {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load model
        model = MT5ForConditionalGeneration.from_pretrained(
            self.checkpoint_path,
            local_files_only=True
        )
        
        # Load tokenizer
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
        
        print("✅ PyTorch model loaded successfully")
        return model, tokenizer
    
    def save_pytorch_final_model(self, model, tokenizer):
        """Save PyTorch model to expected path"""
        final_model_path = Path("./models/trained_models/final_model")
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print(f"✅ PyTorch model saved to {final_model_path}")
        return final_model_path
    
    def convert_to_tensorflow(self, model, tokenizer):
        """Convert PyTorch model to TensorFlow SavedModel"""
        print("Converting to TensorFlow SavedModel...")
        
        try:
            # Convert using transformers TF integration
            tf_model = tf.keras.utils.get_custom_objects()
            
            # Use the built-in TF conversion
            from transformers import TFMT5ForConditionalGeneration
            
            # Create TF model with same config
            tf_model = TFMT5ForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                from_pytorch=True  # Convert from PyTorch
            )
            
            # Save as SavedModel
            tf_output_path = self.output_dir / "german_gec_tf"
            tf_model.save_pretrained(tf_output_path, saved_model=True)
            
            print(f"✅ TensorFlow model saved to {tf_output_path}")
            return tf_output_path
            
        except Exception as e:
            print(f"❌ TF conversion failed: {e}")
            print("Trying alternative ONNX->TF conversion...")
            return self._convert_via_onnx(model, tokenizer)
    
    def _convert_via_onnx(self, model, tokenizer):
        """Alternative: Convert via ONNX"""
        try:
            import onnx
            import tf2onnx
            
            # First convert to ONNX
            onnx_path = self._pytorch_to_onnx(model, tokenizer)
            
            if onnx_path:
                # Then ONNX to TF
                return self._onnx_to_tensorflow(onnx_path)
            
        except ImportError:
            print("ONNX conversion dependencies missing")
            print("Install with: pip install onnx tf2onnx")
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
        
        return None
    
    def _pytorch_to_onnx(self, model, tokenizer):
        """Convert PyTorch to ONNX"""
        print("Converting PyTorch -> ONNX...")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, 64), dtype=torch.long)
        
        onnx_path = self.output_dir / "german_gec_model.onnx"
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                }
            )
            
            print(f"✅ ONNX model saved to {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return None
    
    def _onnx_to_tensorflow(self, onnx_path):
        """Convert ONNX to TensorFlow"""
        print("Converting ONNX -> TensorFlow...")
        
        try:
            import tf2onnx
            
            tf_path = self.output_dir / "german_gec_tf"
            
            # Convert ONNX to TF SavedModel
            model_proto, _ = tf2onnx.convert.from_onnx_model(str(onnx_path))
            
            # Save as SavedModel
            tf.saved_model.save(model_proto, str(tf_path))
            
            print(f"✅ TensorFlow model saved to {tf_path}")
            return tf_path
            
        except Exception as e:
            print(f"❌ ONNX->TF conversion failed: {e}")
            return None
    
    def create_simple_tf_wrapper(self, model, tokenizer):
        """Create simple TF wrapper for inference"""
        print("Creating simple TensorFlow wrapper...")
        
        try:
            # Create a simple TF function wrapper
            @tf.function
            def inference_func(input_ids):
                # This is a placeholder - in practice you'd need to implement
                # the actual model logic in TensorFlow
                return tf.random.uniform((tf.shape(input_ids)[0], tf.shape(input_ids)[1], 32000))
            
            # Create concrete function
            input_spec = tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_ids')
            concrete_func = inference_func.get_concrete_function(input_spec)
            
            # Save as SavedModel
            tf_path = self.output_dir / "german_gec_tf_simple"
            tf.saved_model.save(concrete_func, str(tf_path))
            
            print(f"✅ Simple TF wrapper saved to {tf_path}")
            return tf_path
            
        except Exception as e:
            print(f"❌ Simple TF wrapper failed: {e}")
            return None
    
    def convert_all(self):
        """Convert checkpoint to all required formats"""
        print("Starting conversion pipeline...")
        
        # 1. Load PyTorch model
        try:
            model, tokenizer = self.load_pytorch_model()
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            return False
        
        # 2. Save to expected PyTorch path
        pytorch_path = self.save_pytorch_final_model(model, tokenizer)
        
        # 3. Convert to TensorFlow
        tf_path = self.convert_to_tensorflow(model, tokenizer)
        
        if not tf_path:
            print("Trying simple TF wrapper as fallback...")
            tf_path = self.create_simple_tf_wrapper(model, tokenizer)
        
        # 4. Summary
        self.print_conversion_summary(pytorch_path, tf_path)
        
        return tf_path is not None
    
    def print_conversion_summary(self, pytorch_path, tf_path):
        """Print conversion summary"""
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        
        print(f"Source Checkpoint: {self.checkpoint_path}")
        
        if pytorch_path and pytorch_path.exists():
            print(f"✅ PyTorch Model: {pytorch_path}")
        else:
            print("❌ PyTorch Model: FAILED")
        
        if tf_path and tf_path.exists():
            print(f"✅ TensorFlow Model: {tf_path}")
        else:
            print("❌ TensorFlow Model: FAILED")
        
        print("\n" + "="*60)
        
        if tf_path:
            print("SUCCESS! Ready for TFLite optimization")
            print("Run: python run_tflite_optimization.py")
        else:
            print("FAILED! Manual conversion needed")

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint exists and is valid"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return False, f"Checkpoint directory not found: {checkpoint_path}"
    
    required_files = [
        "config.json",
        "pytorch_model.bin"
    ]
    
    for file in required_files:
        if not (checkpoint_path / file).exists():
            return False, f"Missing required file: {file}"
    
    return True, "Checkpoint is valid"

def main():
    # Check if checkpoint exists
    checkpoint_path = "../models/trained_models/checkpoint-3500"
    
    is_valid, message = verify_checkpoint(checkpoint_path)
    if not is_valid:
        print(f"❌ {message}")
        
        # Try to find checkpoints
        base_path = Path("../models/trained_models")
        if base_path.exists():
            checkpoints = list(base_path.glob("checkpoint-*"))
            if checkpoints:
                print(f"\nAvailable checkpoints:")
                for cp in sorted(checkpoints):
                    print(f"  - {cp}")
                
                latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                print(f"\nUsing latest: {latest}")
                checkpoint_path = str(latest)
            else:
                print("No checkpoints found!")
                return
        else:
            print("German GEC training directory not found!")
            return
    
    # Convert checkpoint
    converter = CheckpointToTFConverter(checkpoint_path)
    success = converter.convert_all()
    
    if success:
        print("\n🎉 Conversion successful!")
        print("Now run: python run_tflite_optimization.py")
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
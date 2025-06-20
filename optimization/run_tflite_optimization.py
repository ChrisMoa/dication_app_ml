import sys
import os
from pathlib import Path

def check_prerequisites():
    """Check if required models exist"""
    required_paths = [
        "./models/converted/tensorflow",  # TensorFlow SavedModel
        "./models/trained_models/final_model"  # Original PyTorch model
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("❌ Missing required models:")
        for path in missing:
            print(f"   - {path}")
        return False
    
    print("✅ All prerequisites found")
    return True

def install_requirements():
    """Install required packages for optimization"""
    import subprocess
    
    packages = [
        "tensorflow>=2.13.0",
        "tensorflow-model-optimization",
        "onnx",
        "tf2onnx",
    ]
    
    print("Installing optimization requirements...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def run_optimization_pipeline():
    """Run complete TFLite optimization pipeline"""
    
    print("\n" + "="*60)
    print("TFLITE OPTIMIZATION PIPELINE")
    print("="*60)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\nPlease run previous conversion steps first!")
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\nFailed to install requirements!")
        return False
    
    # Step 3: TFLite conversion
    print("\nStep 1: Converting to TFLite variants...")
    try:
        from convert_to_tflite import TFLiteConverter
        
        converter = TFLiteConverter("./models/converted/tensorflow")
        tflite_results = converter.convert_all()
        
        print("✅ TFLite conversion complete")
        
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return False
    
    # Step 4: Advanced optimization
    print("\nStep 2: Advanced model optimization...")
    try:
        from optimize_for_mobile import ModelOptimizer
        
        optimizer = ModelOptimizer("./models/converted/tensorflow")
        optimization_results = optimizer.optimize_all()
        
        print("✅ Advanced optimization complete")
        
    except Exception as e:
        print(f"❌ Advanced optimization failed: {e}")
        print("Continuing with basic TFLite models...")
        optimization_results = {}
    
    # Step 5: Performance testing
    print("\nStep 3: Performance benchmarking...")
    try:
        from tflite_inference import benchmark_all_models
        
        benchmark_results = benchmark_all_models()
        
        print("✅ Benchmarking complete")
        print_benchmark_summary(benchmark_results)
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
    
    # Step 6: Mobile asset preparation
    print("\nStep 4: Preparing mobile assets...")
    try:
        from prepare_mobile_assets import MobileAssetPreparer
        
        preparer = MobileAssetPreparer()
        mobile_results = preparer.setup_all()
        
        if mobile_results['status'] == 'ready':
            print("✅ Mobile setup complete")
        else:
            print("⚠️ Mobile setup incomplete")
        
    except Exception as e:
        print(f"❌ Mobile setup failed: {e}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    
    return True

def print_benchmark_summary(results):
    """Print benchmark summary"""
    print("\nPerformance Summary:")
    print("-" * 40)
    
    for model_name, data in results.items():
        if 'error' not in data:
            avg_latency = data.get('avg_latency_ms', 0)
            print(f"{model_name:20} {avg_latency:8.2f}ms")
    
    # Recommendations
    print("\nRecommendations:")
    print("- Use 'dynamic' model for best size/speed balance")
    print("- Use 'int8' model for smallest size (if available)")
    print("- Target <500ms latency for mobile")

def quick_test():
    """Quick functionality test"""
    print("\nRunning quick test...")
    
    try:
        from tflite_inference import TFLiteGECInference
        
        model_path = "./models/mobile/german_gec_dynamic.tflite"
        if not os.path.exists(model_path):
            print(f"Test model not found: {model_path}")
            return False
        
        corrector = TFLiteGECInference(model_path)
        result = corrector.correct_text("Das ist ein fehler.")
        
        print(f"✅ Test successful:")
        print(f"   Original: {result['original']}")
        print(f"   Corrected: {result['corrected']}")
        print(f"   Time: {result['inference_time_ms']:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main optimization runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TFLite Optimization Pipeline')
    parser.add_argument('--test-only', action='store_true', 
                       help='Run quick test only')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip package installation')
    
    args = parser.parse_args()
    
    if args.test_only:
        quick_test()
        return
    
    # Run full optimization pipeline
    success = run_optimization_pipeline()
    
    if success:
        print("\n🎉 Ready for Flutter integration!")
        print("\nNext steps:")
        print("1. cd flutter_gec_app")
        print("2. flutter pub get")
        print("3. flutter run")
        
        # Quick test
        quick_test()
    else:
        print("\n❌ Optimization failed!")
        print("Check error messages above and retry.")

if __name__ == "__main__":
    main()
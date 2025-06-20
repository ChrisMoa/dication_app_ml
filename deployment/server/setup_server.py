# setup_server.py
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def check_requirements():
    """Check if all files exist"""
    required_files = [
        "./models/trained_models/final_model/",
        "convert_to_onnx.py",
        "gec_server.py",
        "deployment/server/requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command(
        f"{sys.executable} -m pip install -r deployment/server/requirements.txt",
        "Installing dependencies"
    )

def convert_model():
    """Convert PyTorch model to ONNX"""
    return run_command(
        f"{sys.executable} convert_to_onnx.py",
        "Converting model to ONNX"
    )

def test_server():
    """Test server functionality"""
    import time
    import requests
    import threading
    from gec_server import app
    import uvicorn
    
    # Start server in background
    def start_server():
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8001/api/v1/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server health check passed")
            
            # Test correction endpoint
            test_data = {"text": "Das ist ein test satz."}
            response = requests.post(
                "http://127.0.0.1:8001/api/v1/correct",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Correction test passed")
                print(f"   Original: {result['original_text']}")
                print(f"   Corrected: {result['corrected_text']}")
                print(f"   Time: {result['processing_time']:.3f}s")
                return True
            else:
                print(f"❌ Correction test failed: {response.status_code}")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Server test failed: {str(e)}")
        return False

def main():
    print("🚀 Setting up German GEC Server...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Convert model
    if not convert_model():
        print("❌ Failed to convert model")
        sys.exit(1)
    
    # Test server
    print("\n🧪 Testing server...")
    if test_server():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start server: python gec_server.py")
        print("2. Test API: curl http://localhost:8000/api/v1/health")
        print("3. Docker build: docker build -t german-gec-server .")
    else:
        print("\n❌ Server test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
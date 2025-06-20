# simple_move_script.py
# First, let's see what files exist and move them properly

import os
import shutil
from pathlib import Path

def list_current_files():
    """List all Python files in current directory"""
    print("📋 Current directory contents:")
    for item in os.listdir('.'):
        if os.path.isfile(item):
            print(f"  📄 {item}")
        elif os.path.isdir(item):
            print(f"  📁 {item}/")
            # List contents of subdirectories
            try:
                for subitem in os.listdir(item):
                    print(f"    📄 {item}/{subitem}")
            except PermissionError:
                print(f"    ❌ Permission denied")

def move_files():
    """Move files to new structure"""
    
    # Ensure directories exist
    directories = [
        "training", "optimization", "deployment/server", 
        "deployment/scripts", "testing"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Define file moves with correct python/ prefix
    file_moves = {
        # Root level files in python/
        "python/train_german_gec_mt5.py": "training/",
        "python/setup_and_run.py": "training/", 
        "python/test_german_gec.py": "testing/",
        "python/german_hybrid_corrector.py": "training/",
        "python/requirements_gec.txt": "./requirements.txt",
        "python/README.md": "./README_python.md",
    }
    
    # Check for subdirectories in python/
    python_subdirs = []
    if os.path.exists("python"):
        for item in os.listdir("python"):
            item_path = os.path.join("python", item)
            if os.path.isdir(item_path):
                python_subdirs.append(item)
                print(f"📁 Found subdirectory: {item_path}")
                
                # List files in subdirectory
                try:
                    for subfile in os.listdir(item_path):
                        subfile_path = os.path.join(item_path, subfile)
                        if os.path.isfile(subfile_path):
                            print(f"  📄 {subfile_path}")
                            
                            # Add server files
                            if item == "02_ForExternalServer":
                                file_moves[subfile_path] = "deployment/server/" if subfile != "deploy.sh" else "deployment/scripts/"
                            
                            # Add optimization files
                            elif item == "03_TFLiteOptimierung":
                                file_moves[subfile_path] = "optimization/"
                                
                except PermissionError:
                    print(f"    ❌ Cannot read {item_path}")
    
    print(f"\n📋 Files to move ({len(file_moves)} total):")
    
    moved_files = []
    missing_files = []
    
    for source, dest_dir in file_moves.items():
        print(f"  {source} → {dest_dir}")
    
    print(f"\n📦 Starting file moves...")
    
    for source, dest_dir in file_moves.items():
        if os.path.exists(source):
            filename = os.path.basename(source)
            
            # Handle special rename for requirements
            if "requirements_gec.txt" in source:
                filename = "requirements.txt"
            elif "README.md" in source and "python/" in source:
                filename = "README_python.md"
                
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                # Ensure destination directory exists
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(source, dest_path)
                print(f"✅ Moved: {source} → {dest_path}")
                moved_files.append((source, dest_path))
            except Exception as e:
                print(f"❌ Failed to move {source}: {e}")
        else:
            print(f"⚠️  Source not found: {source}")
            missing_files.append(source)
    
    return moved_files, missing_files

def main():
    print("🔍 Checking current file structure...")
    list_current_files()
    
    print("\n📦 Moving files to new structure...")
    moved_files, missing_files = move_files()
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully moved: {len(moved_files)} files")
    print(f"⚠️  Missing files: {len(missing_files)} files")
    
    if missing_files:
        print("\n❓ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
    
    print("\n🎯 Next step: Run the migration script to update paths")
    print("   python migration_script.py")

if __name__ == "__main__":
    main()
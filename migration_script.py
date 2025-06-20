# migration_script.py
# Script to update file paths in Python scripts for the new repository structure

import os
import re
from pathlib import Path

class PathMigrator:
    def __init__(self):
        self.path_mappings = {
            # Model paths
            "./german_gec_mt5": "./models/trained_models",
            "../german_gec_mt5": "./models/trained_models", 
            "german_gec_mt5/": "models/trained_models/",
            
            # Output directories
            "./models/german_gec_tf": "./models/converted/tensorflow",
            "./models_mobile/": "./models/mobile/",
            "./models_optimized": "./models/optimized",
            
            # Script directories
            "python/02_ForExternalServer/": "deployment/server/",
            "python/03_TFLiteOptimierung/": "optimization/",
            
            # Configuration files
            "requirements_gec.txt": "requirements.txt",
            "requirements_server.txt": "deployment/server/requirements.txt",
        }
        
        # Files that need path updates (after moving)
        self.files_to_update = [
            # Training scripts
            "training/train_german_gec_mt5.py",
            "training/setup_and_run.py",
            
            # Optimization scripts  
            "optimization/convert_checkpoint_to_tf.py",
            "optimization/run_tflite_optimization.py",
            "optimization/fix_tf_model_signature.py",
            "optimization/prepare_mobile_assets.py",
            
            # Server deployment
            "deployment/server/setup_server.py",
            "deployment/scripts/deploy.sh",
            "deployment/server/gec_server_pytorch.py",
            
            # Testing
            "testing/test_german_gec.py",
        ]

    def update_file_paths(self, file_path):
        """Update paths in a single file"""
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            return
            
        print(f"📝 Updating: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply path mappings
        for old_path, new_path in self.path_mappings.items():
            content = content.replace(old_path, new_path)
        
        # Update specific patterns
        content = self.update_specific_patterns(content, file_path)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
        else:
            print(f"ℹ️  No changes needed: {file_path}")

    def update_specific_patterns(self, content, file_path):
        """Update file-specific patterns"""
        
        # Training scripts - update model output paths
        if "train" in file_path:
            content = re.sub(
                r'output_dir="[^"]*german_gec_mt5[^"]*"',
                'output_dir="./models/trained_models"',
                content
            )
            
        # Optimization scripts - update input paths
        if "optimization" in file_path:
            content = re.sub(
                r'checkpoint_path = "[^"]*german_gec_mt5[^"]*"',
                'checkpoint_path = "./models/trained_models"',
                content
            )
            
        # Deployment scripts - update model references
        if "deploy" in file_path:
            content = content.replace(
                'german_gec_mt5/',
                'models/trained_models/'
            )
            
        # Flutter asset preparation
        if "prepare_mobile_assets" in file_path:
            content = re.sub(
                r'self\.models_path = Path\("[^"]*"\)',
                'self.models_path = Path("./models/mobile")',
                content
            )
            
        return content

    def create_new_directory_structure(self):
        """Create the new directory structure"""
        directories = [
            "data_preparation",
            "training", 
            "optimization",
            "deployment/server",
            "deployment/scripts",
            "testing",
            "models/trained_models",
            "models/converted/tensorflow",
            "models/converted/onnx", 
            "models/mobile",
            "models/optimized",
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created: {directory}")

    def migrate_files(self):
        """Move files to new structure - Skip if already moved"""
        print("📋 Checking if files need to be moved...")
        
        # Check if files are already in the correct locations
        expected_files = [
            "training/train_german_gec_mt5.py",
            "training/setup_and_run.py", 
            "testing/test_german_gec.py",
            "optimization/convert_checkpoint_to_tf.py",
            "deployment/server/gec_server_pytorch.py"
        ]
        
        files_already_moved = all(os.path.exists(f) for f in expected_files)
        
        if files_already_moved:
            print("✅ Files already moved to correct locations")
            return
        
        file_moves = {
            # Training files (in root)
            "train_german_gec_mt5.py": "training/",
            "setup_and_run.py": "training/",
            
            # Optimization files
            "03_TFLiteOptimierung/convert_checkpoint_to_tf.py": "optimization/",
            "03_TFLiteOptimierung/run_tflite_optimization.py": "optimization/", 
            "03_TFLiteOptimierung/fix_tf_model_signature.py": "optimization/",
            "03_TFLiteOptimierung/prepare_mobile_assets.py": "optimization/",
            
            # Server files
            "02_ForExternalServer/setup_server.py": "deployment/server/",
            "02_ForExternalServer/deploy.sh": "deployment/scripts/",
            "02_ForExternalServer/gec_server_pytorch.py": "deployment/server/",
            "02_ForExternalServer/requirements_server.txt": "deployment/server/requirements.txt",
            "02_ForExternalServer/Dockerfile": "deployment/server/",
            
            # Testing files (in root)
            "test_german_gec.py": "testing/",
            
            # Other important files
            "README.md": "./",
            "requirements.txt": "./",
        }
        
        for source, dest_dir in file_moves.items():
            if os.path.exists(source):
                dest_path = os.path.join(dest_dir, os.path.basename(source))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy file (for safety, don't delete original yet)
                import shutil
                shutil.copy2(source, dest_path)
                print(f"📋 Moved: {source} → {dest_path}")

    def update_import_statements(self):
        """Update Python import statements"""
        for file_path in self.files_to_update:
            if os.path.exists(file_path):
                self.update_imports_in_file(file_path)

    def update_imports_in_file(self, file_path):
        """Update imports in a specific file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update relative imports to absolute
        import_updates = {
            "from german_hybrid_corrector import": "from training.german_hybrid_corrector import",
            "from ..training": "from training",
            "import ../": "import ",
        }
        
        original_content = content
        for old_import, new_import in import_updates.items():
            content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"🔗 Updated imports: {file_path}")

    def create_config_file(self):
        """Create configuration file for paths"""
        config_content = '''# config.py
# Configuration file for dictation_app_ml

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
CONVERTED_MODELS_DIR = MODELS_DIR / "converted"
MOBILE_MODELS_DIR = MODELS_DIR / "mobile"
OPTIMIZED_MODELS_DIR = MODELS_DIR / "optimized"

# Default model name (configurable)
DEFAULT_MODEL_NAME = "dictation_gec_model"  # Changed from german_gec_mt5

# Training configuration
TRAINING_CONFIG = {
    "model_name": "google/mt5-small",
    "output_dir": str(TRAINED_MODELS_DIR),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-5,
    "max_length": 64,
}

# Server configuration
SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8001,
    "model_path": str(TRAINED_MODELS_DIR / "final_model"),
}

# Create directories
for directory in [MODELS_DIR, DATA_DIR, TRAINED_MODELS_DIR, 
                  CONVERTED_MODELS_DIR, MOBILE_MODELS_DIR, OPTIMIZED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
'''
        
        with open("config.py", 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("⚙️  Created: config.py")

    def run_migration(self):
        """Run complete migration"""
        print("🚀 Starting Migration to dictation_app_ml")
        print("=" * 50)
        
        # 1. Create new directory structure
        print("\n1. Creating directory structure...")
        self.create_new_directory_structure()
        
        # 2. Create configuration file
        print("\n2. Creating configuration...")
        self.create_config_file()
        
        # 3. Move files
        print("\n3. Moving files...")
        self.migrate_files()
        
        # 4. Update file paths
        print("\n4. Updating file paths...")
        for file_path in self.files_to_update:
            self.update_file_paths(file_path)
        
        # 5. Update imports
        print("\n5. Updating imports...")
        self.update_import_statements()
        
        print("\n✅ Migration completed!")
        print("\nNext steps:")
        print("1. Review updated files")
        print("2. Test scripts with new paths")
        print("3. Remove old files after verification")
        print("4. Update README.md with new structure")

if __name__ == "__main__":
    migrator = PathMigrator()
    migrator.run_migration()
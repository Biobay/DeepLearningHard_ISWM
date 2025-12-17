#!/usr/bin/env python3
"""
Script Python completo per Salad Cloud.
Scarica repo GitHub, dataset Kaggle, esegue training completo.

USAGE SU SALAD:
Command: python3 -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/salad_complete.py').read())"
"""

import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path

# ============ CONFIGURAZIONE ============
REPO_URL = "https://github.com/Biobay/DeepLearningHard_ISWM.git"
KAGGLE_DATASET = "lakshaymiddha/crack-segmentation-dataset"

# IMPORTANTE: Sostituisci con le tue credenziali Kaggle
# Ottienile da: https://www.kaggle.com/settings → API → Create New Token
KAGGLE_USERNAME = "mariomastrulli"
KAGGLE_KEY = "KGAT_08037a2cf26b2f7ffa2612c5b6764b04"

WORKSPACE = "/workspace"
PROJECT_NAME = "DeepLearningHard_ISWM"

# ============================================
# FUNZIONI HELPER
# ============================================

def log(message, emoji=""):
    """Log con timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {emoji} {message}")

def run_cmd(command, description="", show_output=True):
    """Esegue comando shell."""
    if description:
        log(description, "⚙️")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=not show_output,
        text=True
    )
    
    if result.returncode != 0:
        log(f"ERROR: {command}", "❌")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)
    
    return result

def print_section(title):
    """Print sezione con stile."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

# ============================================
# STEP 1: INSTALL SYSTEM DEPENDENCIES
# ============================================

def install_system_deps():
    print_section("📦 INSTALLING SYSTEM DEPENDENCIES")
    
    run_cmd("apt-get update -qq", "Updating apt")
    run_cmd("apt-get install -y git wget curl unzip -qq", "Installing git, wget, curl, unzip")
    
    log("System dependencies installed ✓", "✅")

# ============================================
# STEP 2: CLONE REPOSITORY
# ============================================

def clone_repository():
    print_section("📥 CLONING GITHUB REPOSITORY")
    
    project_path = f"{WORKSPACE}/{PROJECT_NAME}"
    
    os.chdir(WORKSPACE)
    
    if os.path.exists(project_path):
        log("Repository already exists, pulling updates...", "🔄")
        os.chdir(project_path)
        run_cmd("git pull", show_output=False)
    else:
        log(f"Cloning {REPO_URL}", "📦")
        run_cmd(f"git clone {REPO_URL}", show_output=False)
        os.chdir(project_path)
    
    log(f"Repository ready at {project_path} ✓", "✅")
    return project_path

# ============================================
# STEP 3: SETUP KAGGLE CREDENTIALS
# ============================================

def setup_kaggle():
    print_section("🔑 SETTING UP KAGGLE CREDENTIALS")
    
    if KAGGLE_USERNAME == "YOUR_KAGGLE_USERNAME":
        log("ERROR: You need to set KAGGLE_USERNAME and KAGGLE_KEY!", "❌")
        log("Get them from: https://www.kaggle.com/settings", "ℹ️")
        sys.exit(1)
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Write kaggle.json
    kaggle_config = {
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    }
    
    kaggle_file = kaggle_dir / "kaggle.json"
    with open(kaggle_file, 'w') as f:
        json.dump(kaggle_config, f)
    
    # Set permissions
    os.chmod(kaggle_file, 0o600)
    
    log("Kaggle credentials configured ✓", "✅")

# ============================================
# STEP 4: INSTALL PYTHON DEPENDENCIES
# ============================================

def install_python_deps(project_path):
    print_section("🐍 INSTALLING PYTHON DEPENDENCIES")
    
    # Install kaggle CLI
    log("Installing kaggle CLI", "📦")
    run_cmd("pip install --quiet kaggle", show_output=False)
    
    # Install project requirements
    requirements = project_path / "requirements.txt"
    if requirements.exists():
        log("Installing project requirements", "📦")
        run_cmd(f"pip install --quiet -r {requirements}", show_output=False)
    
    log("Python dependencies installed ✓", "✅")

# ============================================
# STEP 5: VERIFY GPU
# ============================================

def verify_gpu():
    print_section("🎮 CHECKING GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            log(f"GPU: {gpu_name}", "🎮")
            log(f"Memory: {gpu_memory:.2f} GB", "💾")
            log("CUDA is available ✓", "✅")
        else:
            log("WARNING: No GPU detected! Training will be slow.", "⚠️")
    
    except ImportError:
        log("PyTorch not installed yet, skipping GPU check", "⚠️")

# ============================================
# STEP 6: DOWNLOAD DATASET FROM KAGGLE
# ============================================

def download_dataset(project_path):
    print_section("📥 DOWNLOADING DATASET FROM KAGGLE")
    
    dataset_path = project_path / "dataset"
    train_images = dataset_path / "train" / "images"
    
    # Check if dataset already exists
    if train_images.exists() and list(train_images.glob("*.jpg")):
        num_images = len(list(train_images.glob("*.jpg")))
        log(f"Dataset already exists ({num_images} images), skipping download", "✓")
        return
    
    log(f"Downloading {KAGGLE_DATASET} (2.1 GB)", "📥")
    os.chdir(project_path)
    
    # Download dataset
    run_cmd(f"kaggle datasets download -d {KAGGLE_DATASET}", show_output=False)
    
    # Find zip file
    zip_files = list(Path(".").glob("*.zip"))
    if not zip_files:
        log("ERROR: No zip file found after download", "❌")
        sys.exit(1)
    
    zip_file = zip_files[0]
    log(f"Extracting {zip_file.name}", "📦")
    
    # Extract
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(project_path)
    
    # Remove zip
    zip_file.unlink()
    
    # Organize dataset structure
    log("Organizing dataset structure", "📁")
    organize_dataset(project_path)
    
    # Verify
    train_count = len(list((dataset_path / "train" / "images").glob("*.jpg")))
    test_count = len(list((dataset_path / "test" / "images").glob("*.jpg")))
    
    log(f"Training images: {train_count}", "✓")
    log(f"Test images: {test_count}", "✓")
    log("Dataset downloaded and organized ✓", "✅")

def organize_dataset(project_path):
    """Organize Kaggle dataset into correct structure."""
    dataset_root = project_path / "dataset"
    dataset_root.mkdir(exist_ok=True)
    
    # Mappings from Kaggle structure to our structure
    mappings = {
        'train_images': dataset_root / 'train' / 'images',
        'train_masks': dataset_root / 'train' / 'masks',
        'test_images': dataset_root / 'test' / 'images',
        'test_masks': dataset_root / 'test' / 'masks',
    }
    
    for src_name, dest_path in mappings.items():
        src_path = project_path / src_name
        
        if src_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))

# ============================================
# STEP 7: SETUP DIRECTORIES
# ============================================

def setup_directories(project_path):
    print_section("📁 CREATING DIRECTORIES")
    
    directories = ['models', 'checkpoints', 'predictions', 'runs']
    
    for dir_name in directories:
        dir_path = project_path / dir_name
        dir_path.mkdir(exist_ok=True)
        log(f"Created {dir_name}/", "📁")
    
    log("Directories created ✓", "✅")

# ============================================
# STEP 8: TRAINING
# ============================================

def run_training(project_path):
    print_section("🚀 STARTING TRAINING (50 EPOCHS)")
    
    os.chdir(project_path)
    
    log("Running train_cloud.py with resume support", "🚀")
    
    result = subprocess.run(
        [sys.executable, "train_cloud.py", "--resume"],
        text=True
    )
    
    if result.returncode != 0:
        log("Training failed!", "❌")
        sys.exit(1)
    
    log("Training completed ✓", "✅")

# ============================================
# STEP 9: INFERENCE
# ============================================

def run_inference(project_path):
    print_section("🔮 RUNNING INFERENCE")
    
    os.chdir(project_path)
    
    log("Generating prediction masks", "🔮")
    
    result = subprocess.run(
        [sys.executable, "inference.py"],
        text=True
    )
    
    if result.returncode != 0:
        log("Inference had some errors, continuing...", "⚠️")
    else:
        log("Inference completed ✓", "✅")

# ============================================
# STEP 10: EVALUATION
# ============================================

def run_evaluation(project_path):
    print_section("📊 RUNNING EVALUATION")
    
    os.chdir(project_path)
    
    log("Calculating metrics (IoU, Dice, F1)", "📊")
    
    result = subprocess.run(
        [sys.executable, "evaluate.py"],
        text=True
    )
    
    if result.returncode != 0:
        log("Evaluation had some errors, continuing...", "⚠️")
    else:
        log("Evaluation completed ✓", "✅")

# ============================================
# STEP 11: SUMMARY
# ============================================

def print_summary(project_path):
    print_section("✅ ALL COMPLETED!")
    
    print("\n📁 Results location:")
    print(f"  {project_path}/models/best_autoencoder.pth")
    print(f"  {project_path}/predictions/")
    print(f"  {project_path}/*.png")
    
    # Model info
    model_path = project_path / "models" / "best_autoencoder.pth"
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"\n📊 Model size: {size_mb:.2f} MB")
    
    # Predictions count
    predictions = list((project_path / "predictions").glob("*.jpg"))
    print(f"🔢 Generated {len(predictions)} prediction masks")
    
    print("\n" + "="*60)

# ============================================
# STEP 12: KEEP ALIVE
# ============================================

def keep_alive(minutes=30):
    print_section(f"⏰ KEEPING CONTAINER ALIVE FOR {minutes} MINUTES")
    
    print("Download your results now from Salad Portal!")
    print("Press Ctrl+C to stop early\n")
    
    try:
        time.sleep(minutes * 60)
    except KeyboardInterrupt:
        print("\n\nStopping early. Goodbye! 👋")

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*60)
    print("  🚀 SALAD CLOUD - CRACK DETECTION TRAINING")
    print("  Fully automated: Clone → Download → Train → Evaluate")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Execute all steps
        install_system_deps()
        project_path = Path(clone_repository())
        setup_kaggle()
        install_python_deps(project_path)
        verify_gpu()
        download_dataset(project_path)
        setup_directories(project_path)
        run_training(project_path)
        run_inference(project_path)
        run_evaluation(project_path)
        print_summary(project_path)
        
        # Calculate duration
        duration = (time.time() - start_time) / 3600
        print(f"\n⏱️  Total duration: {duration:.2f} hours")
        
        # Keep alive for download
        keep_alive(30)
        
    except Exception as e:
        log(f"FATAL ERROR: {e}", "❌")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

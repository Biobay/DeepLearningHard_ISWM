"""
Script Autosufficiente per Salad Cloud.
TUTTO AUTOMATICO: scarica repo, dataset, addestra, salva risultati.

COME USARE SU SALAD:
1. Crea container con immagine: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
2. Command: python -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/auto_salad.py)"
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
import shutil
from pathlib import Path

# ============ CONFIGURAZIONE ============
REPO_URL = "https://github.com/Biobay/DeepLearningHard_ISWM.git"
KAGGLE_DATASET = "lakshaymiddha/crack-segmentation-dataset"  # Dataset Kaggle
KAGGLE_USERNAME = "mariomastrulli"  # Sostituisci con tuo username Kaggle
KAGGLE_KEY = "salad_cloud_user_WQuzJnYHf1ELT1awJ5gjpBU7JKKwITxrYcRwcAsYsfb4AY1WNY"  # Sostituisci con tua API key Kaggle

WORKSPACE = "/workspace"
PROJECT_NAME = "DeepLearningHard_ISWM"

# ============ FUNZIONI HELPER ============

def run_command(cmd, description):
    """Esegue comando shell con logging."""
    print(f"\n{'='*60}")
    print(f"⚙️  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ERROR: {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    return result

def install_system_deps():
    """Installa dipendenze sistema."""
    print("\n🔧 Installing system dependencies...")
    run_command(
        "apt-get update && apt-get install -y git wget curl",
        "System packages"
    )

def clone_repository():
    """Clona repository GitHub."""
    print("\n📦 Cloning repository...")
    project_path = f"{WORKSPACE}/{PROJECT_NAME}"
    
    if os.path.exists(project_path):
        print("Repository già esistente, pull updates...")
        os.chdir(project_path)
        run_command("git pull", "Git pull")
    else:
        os.chdir(WORKSPACE)
        run_command(f"git clone {REPO_URL}", "Git clone")
    
    return project_path

def setup_kaggle_credentials():
    """Configura credenziali Kaggle."""
    print("\n🔑 Setting up Kaggle credentials...")
    
    # Crea directory kaggle
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Crea kaggle.json
    kaggle_json = {
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    }
    
    kaggle_file = f"{kaggle_dir}/kaggle.json"
    import json
    with open(kaggle_file, 'w') as f:
        json.dump(kaggle_json, f)
    
    # Set permissions (richiesto da Kaggle)
    os.chmod(kaggle_file, 0o600)
    
    print("✓ Kaggle credentials configurate")

def download_dataset(project_path):
    """Scarica dataset da Kaggle."""
    print("\n📥 Downloading dataset from Kaggle...")
    
    # Verifica credenziali
    if KAGGLE_USERNAME == "YOUR_KAGGLE_USERNAME" or KAGGLE_KEY == "YOUR_KAGGLE_API_KEY":
        print("⚠️  WARNING: Devi configurare KAGGLE_USERNAME e KAGGLE_KEY!")
        print("Vedi: https://www.kaggle.com/docs/api")
        print("\nPer ora creo struttura vuota per test...")
        os.makedirs(f"{project_path}/dataset/train/images", exist_ok=True)
        os.makedirs(f"{project_path}/dataset/test/images", exist_ok=True)
        os.makedirs(f"{project_path}/dataset/train/masks", exist_ok=True)
        os.makedirs(f"{project_path}/dataset/test/masks", exist_ok=True)
        return
    
    # Installa kaggle CLI
    run_command("pip install kaggle", "Install Kaggle CLI")
    
    # Setup credenziali
    setup_kaggle_credentials()
    
    # Download dataset
    os.chdir(project_path)
    run_command(
        f"kaggle datasets download -d {KAGGLE_DATASET}",
        "Download from Kaggle"
    )
    
    # Trova file zip scaricato
    zip_file = None
    for f in os.listdir(project_path):
        if f.endswith('.zip'):
            zip_file = f
            break
    
    if not zip_file:
        print("❌ File zip non trovato!")
        sys.exit(1)
    
    # Estrai
    print("\n📦 Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(project_path)
    
    os.remove(zip_file)
    print("✓ Dataset estratto")
    
    # Il dataset Kaggle ha struttura diversa, organizziamolo
    organize_kaggle_dataset(project_path)
    
    # Verifica
    train_path = f"{project_path}/dataset/train/images"
    if os.path.exists(train_path):
        num_images = len([f for f in os.listdir(train_path) if f.endswith('.jpg')])
        print(f"✓ Trovate {num_images} immagini di training")
    else:
        print("⚠️  Struttura dataset diversa da attesa, verifica manualmente")

def organize_kaggle_dataset(project_path):
    """Organizza dataset Kaggle nella struttura corretta."""
    print("\n📁 Organizing dataset structure...")
    
    # Il dataset Kaggle ha questa struttura:
    # train_images/ train_masks/ test_images/ test_masks/
    # Dobbiamo creare: dataset/train/images, dataset/train/masks, etc.
    
    dataset_root = f"{project_path}/dataset"
    os.makedirs(dataset_root, exist_ok=True)
    
    # Mappatura directory Kaggle -> nostra struttura
    mappings = {
        'train_images': 'dataset/train/images',
        'train_masks': 'dataset/train/masks',
        'test_images': 'dataset/test/images',
        'test_masks': 'dataset/test/masks',
    }
    
    for kaggle_dir, target_dir in mappings.items():
        kaggle_path = f"{project_path}/{kaggle_dir}"
        target_path = f"{project_path}/{target_dir}"
        
        if os.path.exists(kaggle_path):
            print(f"Moving {kaggle_dir} -> {target_dir}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.move(kaggle_path, target_path)
    
    print("✓ Dataset organizzato")

def install_python_deps(project_path):
    """Installa dipendenze Python."""
    print("\n🐍 Installing Python dependencies...")
    requirements = f"{project_path}/requirements.txt"
    run_command(
        f"pip install --no-cache-dir -r {requirements}",
        "Python packages"
    )

def verify_gpu():
    """Verifica GPU disponibile."""
    print("\n🎮 Checking GPU...")
    result = subprocess.run(
        ["python", "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else 'N/A')"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if "CUDA: False" in result.stdout:
        print("⚠️  WARNING: GPU non disponibile, training sarà lento!")

def setup_directories(project_path):
    """Crea directory necessarie."""
    print("\n📁 Setting up directories...")
    dirs = ['models', 'checkpoints', 'predictions', 'runs']
    for d in dirs:
        os.makedirs(f"{project_path}/{d}", exist_ok=True)
    print("✓ Directories create")

def run_training(project_path):
    """Esegue training."""
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    os.chdir(project_path)
    
    # Usa train_cloud.py con resume support
    result = subprocess.run(
        ["python", "train_cloud.py", "--resume"],
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Training fallito!")
        sys.exit(1)
    
    print("\n✓ Training completato!")

def run_inference(project_path):
    """Esegue inference."""
    print("\n" + "="*60)
    print("🔮 RUNNING INFERENCE")
    print("="*60)
    
    os.chdir(project_path)
    result = subprocess.run(["python", "inference.py"], text=True)
    
    if result.returncode != 0:
        print("⚠️  Inference parzialmente fallita, continuo...")
    else:
        print("\n✓ Inference completata!")

def run_evaluation(project_path):
    """Esegue evaluation."""
    print("\n" + "="*60)
    print("📊 RUNNING EVALUATION")
    print("="*60)
    
    os.chdir(project_path)
    result = subprocess.run(["python", "evaluate.py"], text=True)
    
    if result.returncode != 0:
        print("⚠️  Evaluation parzialmente fallita, continuo...")
    else:
        print("\n✓ Evaluation completata!")

def upload_results_to_github(project_path):
    """Push risultati su GitHub (opzionale)."""
    print("\n📤 Uploading results to GitHub...")
    
    os.chdir(project_path)
    
    # Configura git
    run_command(
        'git config --global user.email "salad@cloud.com"',
        "Git config email"
    )
    run_command(
        'git config --global user.name "Salad Training Bot"',
        "Git config name"
    )
    
    # Add results
    subprocess.run("git add models/*.pth predictions/*.jpg *.png", shell=True)
    subprocess.run('git commit -m "Training results from Salad Cloud"', shell=True)
    
    # Push (richiede token o SSH key configurata)
    result = subprocess.run("git push", shell=True, capture_output=True)
    
    if result.returncode != 0:
        print("⚠️  Push fallito (normale se non hai configurato auth)")
        print("Scarica risultati manualmente dal container")
    else:
        print("✓ Risultati pushati su GitHub!")

def keep_alive(minutes=10):
    """Mantiene container vivo per scaricare risultati."""
    import time
    print(f"\n⏰ Container rimarrà vivo per {minutes} minuti")
    print("Scarica i risultati ora:")
    print("  - models/best_autoencoder.pth")
    print("  - predictions/")
    print("  - results_visualization.png")
    time.sleep(minutes * 60)

# ============ MAIN ============

def main():
    """Entry point principale."""
    print("\n" + "="*60)
    print("🎯 SALAD CLOUD - AUTO TRAINING SCRIPT")
    print("="*60)
    
    try:
        # 1. Setup sistema
        install_system_deps()
        
        # 2. Clone repo
        project_path = clone_repository()
        
        # 3. Download dataset
        download_dataset(project_path)
        
        # 4. Installa dipendenze Python
        install_python_deps(project_path)
        
        # 5. Verifica GPU
        verify_gpu()
        
        # 6. Setup directories
        setup_directories(project_path)
        
        # 7. Training
        run_training(project_path)
        
        # 8. Inference
        run_inference(project_path)
        
        # 9. Evaluation
        run_evaluation(project_path)
        
        # 10. Upload risultati (opzionale)
        # upload_results_to_github(project_path)
        
        # 11. Mostra risultati
        print("\n" + "="*60)
        print("✅ TUTTO COMPLETATO!")
        print("="*60)
        print("\n📁 Risultati disponibili in:")
        print(f"  {project_path}/models/best_autoencoder.pth")
        print(f"  {project_path}/predictions/")
        print(f"  {project_path}/*.png")
        
        # 12. Mantieni vivo per download
        keep_alive(minutes=15)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

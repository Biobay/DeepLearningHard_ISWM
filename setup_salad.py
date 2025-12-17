"""
Script di setup per ambiente Salad Cloud.
Configura l'ambiente, scarica dataset e prepara per training remoto.
"""

import os
import sys
import subprocess
import argparse

def install_dependencies():
    """Installa tutte le dipendenze necessarie."""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("✓ Dependencies installed")

def setup_directories():
    """Crea le directory necessarie per modelli e output."""
    directories = [
        'models',
        'predictions',
        'runs',
        'checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dataset():
    """Verifica che il dataset sia presente."""
    train_path = 'dataset/train/images'
    test_path = 'dataset/test/images'
    
    if not os.path.exists(train_path):
        print(f"ERROR: Dataset non trovato in {train_path}")
        print("Devi montare o scaricare il dataset prima!")
        return False
    
    train_images = len([f for f in os.listdir(train_path) if f.endswith('.jpg')])
    test_images = len([f for f in os.listdir(test_path) if f.endswith('.jpg')])
    
    print(f"✓ Dataset trovato:")
    print(f"  - Train images: {train_images}")
    print(f"  - Test images: {test_images}")
    
    return True

def check_gpu():
    """Verifica disponibilità GPU."""
    import torch
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU disponibile: {device_name}")
        print(f"  - Memory: {memory:.2f} GB")
        return True
    else:
        print("⚠ WARNING: GPU non disponibile, training sarà lento!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup per Salad Cloud')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency installation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SETUP AMBIENTE SALAD CLOUD")
    print("="*60 + "\n")
    
    # 1. Installa dipendenze
    if not args.skip_deps:
        install_dependencies()
    
    # 2. Crea directory
    setup_directories()
    
    # 3. Verifica GPU
    check_gpu()
    
    # 4. Verifica dataset
    dataset_ok = check_dataset()
    
    print("\n" + "="*60)
    if dataset_ok:
        print("SETUP COMPLETATO! Puoi procedere con:")
        print("  python train_cloud.py")
    else:
        print("SETUP INCOMPLETO: dataset mancante!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

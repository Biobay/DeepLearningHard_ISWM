#!/bin/bash
# Script completo per Salad Cloud
# Clona repo, scarica dataset da Kaggle, fa training completo

set -e  # Exit on error

echo "=============================================="
echo "🚀 SALAD CLOUD - CRACK DETECTION TRAINING"
echo "=============================================="

# ========== CONFIGURAZIONE ==========
REPO_URL="https://github.com/Biobay/DeepLearningHard_ISWM.git"
KAGGLE_DATASET="lakshaymiddha/crack-segmentation-dataset"

# IMPORTANTE: Sostituisci con le tue credenziali Kaggle!
# Ottienile da: https://www.kaggle.com/settings → API → Create New Token
KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
KAGGLE_KEY="YOUR_KAGGLE_KEY"

WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/DeepLearningHard_ISWM"

# ========== 1. SETUP SISTEMA ==========
echo ""
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y git wget curl unzip -qq

# ========== 2. CLONE REPOSITORY ==========
echo ""
echo "📥 Cloning repository..."
cd $WORKSPACE
if [ -d "$PROJECT_DIR" ]; then
    echo "Repository già esistente, pulling updates..."
    cd $PROJECT_DIR
    git pull
else
    git clone $REPO_URL
    cd $PROJECT_DIR
fi

# ========== 3. SETUP KAGGLE CREDENTIALS ==========
echo ""
echo "🔑 Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{
  "username": "$KAGGLE_USERNAME",
  "key": "$KAGGLE_KEY"
}
EOF
chmod 600 ~/.kaggle/kaggle.json

# ========== 4. INSTALL PYTHON DEPENDENCIES ==========
echo ""
echo "🐍 Installing Python dependencies..."
pip install --no-cache-dir -q kaggle
pip install --no-cache-dir -q -r requirements.txt

# ========== 5. VERIFY GPU ==========
echo ""
echo "🎮 Checking GPU..."
python3 << 'PYEOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  WARNING: No GPU detected!")
PYEOF

# ========== 6. DOWNLOAD DATASET FROM KAGGLE ==========
echo ""
echo "📥 Downloading dataset from Kaggle (2.1 GB)..."
cd $PROJECT_DIR

# Check if dataset already exists
if [ -d "dataset/train/images" ] && [ "$(ls -A dataset/train/images)" ]; then
    echo "✓ Dataset già presente, skip download"
else
    echo "Downloading from Kaggle..."
    kaggle datasets download -d $KAGGLE_DATASET
    
    # Find the zip file
    ZIP_FILE=$(ls *.zip 2>/dev/null | head -1)
    
    if [ -z "$ZIP_FILE" ]; then
        echo "❌ ERROR: Zip file not found!"
        exit 1
    fi
    
    echo "Extracting $ZIP_FILE..."
    unzip -q $ZIP_FILE
    rm $ZIP_FILE
    
    # Organize dataset structure
    echo "Organizing dataset structure..."
    mkdir -p dataset/train dataset/test
    
    # Move folders if they exist with different names
    [ -d "train_images" ] && mv train_images dataset/train/images
    [ -d "train_masks" ] && mv train_masks dataset/train/masks
    [ -d "test_images" ] && mv test_images dataset/test/images
    [ -d "test_masks" ] && mv test_masks dataset/test/masks
    
    echo "✓ Dataset downloaded and organized"
fi

# Verify dataset
TRAIN_COUNT=$(find dataset/train/images -name "*.jpg" | wc -l)
TEST_COUNT=$(find dataset/test/images -name "*.jpg" | wc -l)
echo "✓ Training images: $TRAIN_COUNT"
echo "✓ Test images: $TEST_COUNT"

# ========== 7. SETUP DIRECTORIES ==========
echo ""
echo "📁 Creating directories..."
mkdir -p models checkpoints predictions runs

# ========== 8. TRAINING ==========
echo ""
echo "=============================================="
echo "🚀 STARTING TRAINING (50 epochs)"
echo "=============================================="
python3 train_cloud.py --resume

# ========== 9. INFERENCE ==========
echo ""
echo "=============================================="
echo "🔮 RUNNING INFERENCE"
echo "=============================================="
python3 inference.py

# ========== 10. EVALUATION ==========
echo ""
echo "=============================================="
echo "📊 RUNNING EVALUATION"
echo "=============================================="
python3 evaluate.py

# ========== 11. SUMMARY ==========
echo ""
echo "=============================================="
echo "✅ ALL COMPLETED!"
echo "=============================================="
echo ""
echo "📁 Results location:"
echo "  - Model: $PROJECT_DIR/models/best_autoencoder.pth"
echo "  - Predictions: $PROJECT_DIR/predictions/"
echo "  - Visualizations: $PROJECT_DIR/*.png"
echo ""
echo "📊 Model info:"
ls -lh models/best_autoencoder.pth 2>/dev/null || echo "  Model file not found"
echo ""
echo "🔢 Predictions count:"
PRED_COUNT=$(find predictions -name "*.jpg" 2>/dev/null | wc -l)
echo "  Generated $PRED_COUNT prediction masks"
echo ""

# ========== 12. KEEP ALIVE ==========
echo "⏰ Keeping container alive for 30 minutes for result download..."
echo "   Download files now from Salad Portal!"
echo ""
echo "Press Ctrl+C to stop early"
sleep 1800  # 30 minutes

echo ""
echo "Container shutting down. Goodbye! 👋"

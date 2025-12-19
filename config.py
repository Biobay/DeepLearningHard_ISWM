import os

# --- DYNAMIC PATH CONFIGURATION ---
# Get the directory where config.py is located (Project Root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define Dataset Root relative to Project Root
# EXPECTED STRUCTURE: project_root/dataset/train/...
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

TRAIN_IMAGES_PATH = os.path.join(DATASET_DIR, 'train', 'images')
TRAIN_MASKS_PATH = os.path.join(DATASET_DIR, 'train', 'masks')
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, 'test', 'images')
TEST_MASKS_PATH = os.path.join(DATASET_DIR, 'test', 'masks')

# OUTPUT DIRS
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models')
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, 'predictions')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')

# HYPERPARAMETERS
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
LATENT_DIM = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
ANOMALY_THRESHOLD = 0.15 
DEVICE = 'cuda' 

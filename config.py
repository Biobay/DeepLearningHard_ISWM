import os

# PATHS (Ensure these match your Kaggle/Colab setup)
# If using KaggleHub download:
DATASET_DIR = '/kaggle/input/crack-segmentation-dataset/crack_segmentation_dataset'
# If using GitHub/Local:
# DATASET_DIR = os.path.join(os.getcwd(), 'dataset')

TRAIN_IMAGES_PATH = os.path.join(DATASET_DIR, 'train', 'images')
TRAIN_MASKS_PATH = os.path.join(DATASET_DIR, 'train', 'masks')
TEST_IMAGES_PATH = os.path.join(DATASET_DIR, 'test', 'images')
TEST_MASKS_PATH = os.path.join(DATASET_DIR, 'test', 'masks')

# OUTPUTS
PROJECT_ROOT = os.getcwd()
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models')
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, 'predictions')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')

# HYPERPARAMETERS
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
LATENT_DIM = 2048  # High capacity for sharp texture
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = 'cuda'

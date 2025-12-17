# Configurazione parametri del progetto

# Percorsi dataset
TRAIN_IMAGES_PATH = 'dataset/train/images/'
TEST_IMAGES_PATH = 'dataset/test/images/'
TEST_MASKS_PATH = 'dataset/test/masks/'

# Parametri immagini
IMAGE_SIZE = (128, 128)  # Ridimensionamento per velocizzare training
INPUT_CHANNELS = 3  # RGB
BATCH_SIZE = 32
NUM_EPOCHS = 50

# Parametri architettura autoencoder
LATENT_DIM = 128  # Dimensione del bottleneck (latent space)

# Parametri training
LEARNING_RATE = 0.001
DEVICE = 'cuda'  # Cambierà automaticamente a 'cpu' se GPU non disponibile

# Parametri anomaly detection
ANOMALY_THRESHOLD = 0.3  # Threshold per binarizzare la differenza (regolabile dopo esperimenti)

# Salvataggio modelli
MODEL_SAVE_PATH = 'models/'
CHECKPOINT_PATH = 'models/autoencoder_checkpoint.pth'
BEST_MODEL_PATH = 'models/best_autoencoder.pth'

# Output predizioni
PREDICTIONS_PATH = 'predictions/'

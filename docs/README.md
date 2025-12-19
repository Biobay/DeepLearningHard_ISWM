# DeepLearningHard_ISWM
Project Aim: hard - image segmentation without mask 5.0 at most https://www.kaggle.com/competitions/dl-2025-project-2-pro

So, we are using an Unsupervised Anomaly Detection approach via a Convolutional Autoencoder (CAE). We Trained a Neural network to become an expert at reconstructing perfect, clean asphalt. When we show it a crack later (inference), It won't know how to reconstruct the crack because it never learned it. It will try to "fix" the crack (reconstruct it as smooth pavement).

Our main idea is When we subtract the Reconstructed Image from the Original Image, the "difference" (residual) will be the crack itself.

| File           | Status       | Evaluation / Functionality |
|----------------|-------------|----------------------------|
| dataset.py     | Done (Pro)  | **Critical Update:** Implements Unsupervised Data Filtering. It automatically scans mask files and excludes any image containing a crack (`mask.sum() > 0`) from the training set. This ensures the Autoencoder only learns **"Healthy" asphalt features**. |
| train.py       | Done        | Implements the training loop using **CombinedLoss (MSE + SSIM)**. • **MSE:** Ensures pixel-level color accuracy. • **SSIM:** Preserves structural texture (prevents blurry asphalt), which is crucial for anomaly detection. |
| inference.py   | Done (Pro)  | Dual-Output Logic: 1. **Segmentation:** Generates the "Difference Map" (`...`). |
| model.py       | Done        | Standard Convolutional Autoencoder (CAE) architecture. Compresses images to a latent dimension (currently 128) to force feature learning, then upsamples to reconstruct the original input. |
| config.py      | Done        | Centralized Configuration. Uses relative paths (`os.getcwd()`) so the code runs on any machine (Colab, Local, Cloud) without changing strings. Manages hyperparameters (`ANOMALY_THRESHOLD`, `LR`, `EPOCHS`). |
| evaluate.py    | Done        | Optimization script. Validates the model against the test set. Tests multiple thresholds (e.g., 0.05 to 0.5) to find the optimal value that maximizes the Dice Score (F1), replacing manual guessing. |
| docs/          | Done        | Contains project documentation (`README.md`, `KAGGLE_SETUP.md`) moved from root to keep the repository clean and professional. |


## Some Initial Predictions:
<img width="982" height="405" alt="image" src="https://github.com/user-attachments/assets/e8497d25-9865-4fce-9d45-ae9122a4ea11" />

--> The "Pred" is too noisy (as of 18/12/25) The final image on the right (Prediction) is a mess. It sees the crack, but it also sees "static" (noise) everywhere else.

## Some Loss functions used:
### Mean Squared Error (MSE): 
For Pixel colors in the reconstructed image match the original.

### Structural Similarity Index (SSIM)
SSIM evaluates whether the **structural information** (such as asphalt texture) is preserved rather than relying solely on raw pixel values.  
This allows the model to ignore lighting variations while detecting **physical cracks**.

### Anomaly Score (Inference)

**Binary Mask:**
  Mask = 1 if M_anom > T  
  Mask = 0 otherwise


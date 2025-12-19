# DeepLearningHard_ISWM
Project Aim: hard - image segmentation without mask 5.0 at most https://www.kaggle.com/competitions/dl-2025-project-2-pro

So, we are using an Unsupervised Anomaly Detection approach via a Convolutional Autoencoder (CAE). We Trained a Neural network to become an expert at reconstructing perfect, clean asphalt. When we show it a crack later (inference), It won't know how to reconstruct the crack because it never learned it. It will try to "fix" the crack (reconstruct it as smooth pavement).

Our main idea is When we subtract the Reconstructed Image from the Original Image, the "difference" (residual) will be the crack itself.

| File         | Status  | Evaluation                                                                                                                                                                                                   |
|--------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model.py     | Done    | Implements a standard ConvAutoencoder. It compresses the image to a "Latent Space" and expands it back.                                                                                                      |
| Train.py     | Done    | Sets up the training loop using  CombinedLoss (MSE + SSIM). This is a smart choice because SSIM (Structural Similarity) prevents the reconstruction from becoming too blurry, which is common with just MSE. |
| Inference.py | Done    | Implements the logic:  Anomaly Map = abs(Input - Output). It also includes morphological post-processing (opening/closing) to clean up noise.                                                                |
| Dataset.py   | Done    | Handles loading images (implied).                                                                                                                                                                            |
| Evaluate.py  | Pending |                                                                                                      |

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


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


Some Loss functions used:
### Mean Squared Error (MSE)

\[
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2
\]

Ensures pixel colors match between the input and reconstructed image.

---

### Structural Similarity Index (SSIM)

SSIM checks whether the **structural information** (e.g., texture of the asphalt) is preserved rather than comparing raw pixel values.  
This helps the model remain robust to lighting changes while still detecting **physical defects such as cracks**.

---

### Anomaly Score (Inference)

\[
M_{\text{anom}} = \lvert X_{\text{input}} - X_{\text{reconstructed}} \rvert
\]

\[
\text{Mask} =
\begin{cases}
1 & \text{if } M_{\text{anom}} > T \\
0 & \text{otherwise}
\end{cases}
\]

Pixels with anomaly scores above the threshold \(T\) are classified as anomalous.


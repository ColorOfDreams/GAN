"""DeepSMOTE tầng 2: train autoencoder + SMOTE trong latent space + decode + ghép.

Input: dataset mất cân bằng do `src/preprocessing/create_imbalanced_dataset.py` tạo
(tầng 1). Output: dataset đã balance (ảnh thật + ảnh synthetic).
"""

"""
Data Augmentation para Séries Temporais Financeiras
"""

import numpy as np
import pandas as pd
from loguru import logger


class TimeSeriesAugmenter:
    """
    Técnicas de Data Augmentation para trading.
    
    Aumenta dataset sem overfitting:
    - Noise injection
    - Temporal flip
    - MixUp
    """
    
    def __init__(self, noise_level=0.01, use_temporal_flip=True, use_mixup=False):
        self.noise_level = noise_level
        self.use_temporal_flip = use_temporal_flip
        self.use_mixup = use_mixup
    
    def augment(self, X, y):
        """
        Aplica augmentations no dataset.
        
        Args:
            X: Features (n_samples, n_features) ou (n_samples, seq_len, n_features)
            y: Labels (n_samples,)
        
        Returns:
            X_augmented, y_augmented (dataset aumentado)
        """
        X_aug_list = [X]
        y_aug_list = [y]
        
        # 1. Noise Injection
        if self.noise_level > 0:
            X_noisy = self._add_noise(X, self.noise_level)
            X_aug_list.append(X_noisy)
            y_aug_list.append(y)
            logger.info(f"✓ Noise injection aplicado ({self.noise_level*100}%)")
        
        # 2. Temporal Flip (apenas para sequências)
        if self.use_temporal_flip and len(X.shape) == 3:
            X_flipped, y_flipped = self._temporal_flip(X, y)
            X_aug_list.append(X_flipped)
            y_aug_list.append(y_flipped)
            logger.info("✓ Temporal flip aplicado")
        
        # 3. MixUp (mixing between samples)
        if self.use_mixup:
            X_mixed, y_mixed = self._mixup(X, y, alpha=0.2)
            X_aug_list.append(X_mixed)
            y_aug_list.append(y_mixed)
            logger.info("✓ MixUp aplicado")
        
        # Concatenar todos
        if len(X.shape) == 2:
            X_augmented = np.vstack(X_aug_list)
        else:  # 3D (sequences)
            X_augmented = np.concatenate(X_aug_list, axis=0)
        
        y_augmented = np.concatenate(y_aug_list, axis=0)
        
        logger.info(f"Dataset aumentado: {X.shape} -> {X_augmented.shape}")
        
        return X_augmented, y_augmented
    
    def _add_noise(self, X, noise_level):
        """Adiciona ruído gaussiano"""
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise * np.std(X, axis=0)  # Escala ruído por std de cada feature
    
    def _temporal_flip(self, X, y):
        """Inverte ordem temporal das sequências"""
        # Inverter eixo temporal (axis=1)
        X_flipped = X[:, ::-1, :]
        
        # Inverter labels (LONG -> SHORT, SHORT -> LONG)
        y_flipped = 1 - y
        
        return X_flipped, y_flipped
    
    def _mixup(self, X, y, alpha=0.2):
        """MixUp: mistura linear entre pares de samples"""
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        # Lambda de distribuição Beta
        lam = np.random.beta(alpha, alpha, n_samples)
        
        if len(X.shape) == 2:
            lam = lam[:, np.newaxis]
        else:  # 3D
            lam = lam[:, np.newaxis, np.newaxis]
        
        # Mix
        X_mixed = lam * X + (1 - lam) * X[indices]
        y_mixed = lam.flatten() * y + (1 - lam.flatten()) * y[indices]
        
        return X_mixed, y_mixed

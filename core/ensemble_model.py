"""
Modelo Ensemble: Deep Learning + XGBoost/LightGBM
Combina as forças de ambos modelos para melhor performance

Estratégias de ensemble:
1. Voting: Média ponderada das probabilidades
2. Stacking: Meta-learner em cima das predições
3. Boosting: Sequencial com correção de erros
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Optional, Tuple
import pickle
from pathlib import Path

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from core.deep_learning_model import DeepScalpingModel, AttentionLayer


class EnsembleModel:
    """
    Modelo Ensemble que combina Deep Learning com XGBoost ou LightGBM.

    Métodos de ensemble:
    - weighted_average: Média ponderada (default)
    - stacking: Meta-learner logistic regression
    - max_confidence: Escolhe predição com maior confiança
    """

    def __init__(
        self,
        dl_model_path: Optional[str] = None,
        tree_model_path: Optional[str] = None,
        tree_type: str = 'xgboost',  # 'xgboost' ou 'lightgbm'
        ensemble_method: str = 'weighted_average',
        dl_weight: float = 0.6,  # 60% DL, 40% tree model
        config: Optional[Dict] = None
    ):
        self.dl_model_path = dl_model_path
        self.tree_model_path = tree_model_path
        self.tree_type = tree_type
        self.ensemble_method = ensemble_method
        self.dl_weight = dl_weight
        self.tree_weight = 1 - dl_weight
        self.config = config or {}

        # Models
        self.dl_model = None
        self.tree_model = None
        self.scaler = None

        # Meta-learner (para stacking)
        self.meta_learner = None

    def load_models(self):
        """
        Carrega os modelos treinados.
        """
        logger.info("Carregando modelos para ensemble...")

        # Carregar Deep Learning model
        if self.dl_model_path and TENSORFLOW_AVAILABLE:
            logger.info(f"Carregando DL model de: {self.dl_model_path}")
            self.dl_model = DeepScalpingModel(self.config)
            self.dl_model.load(self.dl_model_path)
            logger.info("✓ DL model carregado")
        else:
            logger.warning("DL model não disponível")

        # Carregar Tree model
        if self.tree_model_path:
            logger.info(f"Carregando {self.tree_type} model de: {self.tree_model_path}")

            with open(self.tree_model_path, 'rb') as f:
                saved_data = pickle.load(f)

            self.tree_model = saved_data['model']
            logger.info(f"✓ {self.tree_type} model carregado")
        else:
            logger.warning(f"{self.tree_type} model não disponível")

    def predict(
        self,
        X: np.ndarray,
        sequence_length: int = 20
    ) -> np.ndarray:
        """
        Faz predições usando ensemble.

        Args:
            X: Features (n_samples, n_features)
            sequence_length: Comprimento da sequência para DL

        Returns:
            Probabilidades ensemble (n_samples,)
        """
        predictions = []

        # Predições DL
        if self.dl_model is not None:
            logger.debug("Gerando predições DL...")
            dl_pred = self.dl_model.predict(X, sequence_length=sequence_length)
            predictions.append(('dl', dl_pred))

        # Predições Tree model
        if self.tree_model is not None:
            logger.debug(f"Gerando predições {self.tree_type}...")

            # Tree model precisa remover sequências
            # (DL retorna n_samples - sequence_length predições)
            X_tree = X[sequence_length:]

            if self.tree_type == 'xgboost':
                tree_pred = self.tree_model.predict_proba(X_tree)[:, 1]
            elif self.tree_type == 'lightgbm':
                tree_pred = self.tree_model.predict_proba(X_tree)[:, 1]
            else:
                raise ValueError(f"Tree type desconhecido: {self.tree_type}")

            predictions.append(('tree', tree_pred))

        # Combinar predições
        if len(predictions) == 0:
            raise ValueError("Nenhum modelo disponível para ensemble!")

        if len(predictions) == 1:
            logger.warning("Apenas 1 modelo disponível, retornando predição única")
            return predictions[0][1]

        # Ensemble
        ensemble_pred = self._combine_predictions(predictions)

        return ensemble_pred

    def _combine_predictions(
        self,
        predictions: list
    ) -> np.ndarray:
        """
        Combina predições de múltiplos modelos.

        Args:
            predictions: Lista de tuplas (nome, pred_array)

        Returns:
            Predições combinadas
        """
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average(predictions)

        elif self.ensemble_method == 'max_confidence':
            return self._max_confidence(predictions)

        elif self.ensemble_method == 'stacking':
            return self._stacking(predictions)

        else:
            raise ValueError(f"Ensemble method desconhecido: {self.ensemble_method}")

    def _weighted_average(self, predictions: list) -> np.ndarray:
        """
        Média ponderada das predições.
        """
        dl_pred = None
        tree_pred = None

        for name, pred in predictions:
            if name == 'dl':
                dl_pred = pred
            elif name == 'tree':
                tree_pred = pred

        if dl_pred is not None and tree_pred is not None:
            # Ambos disponíveis
            ensemble = self.dl_weight * dl_pred + self.tree_weight * tree_pred
        elif dl_pred is not None:
            ensemble = dl_pred
        else:
            ensemble = tree_pred

        return ensemble

    def _max_confidence(self, predictions: list) -> np.ndarray:
        """
        Escolhe a predição com maior confiança (mais distante de 0.5).
        """
        dl_pred = None
        tree_pred = None

        for name, pred in predictions:
            if name == 'dl':
                dl_pred = pred
            elif name == 'tree':
                tree_pred = pred

        if dl_pred is not None and tree_pred is not None:
            # Calcular confiança (distância de 0.5)
            dl_conf = np.abs(dl_pred - 0.5)
            tree_conf = np.abs(tree_pred - 0.5)

            # Escolher predição com maior confiança
            ensemble = np.where(dl_conf > tree_conf, dl_pred, tree_pred)
        elif dl_pred is not None:
            ensemble = dl_pred
        else:
            ensemble = tree_pred

        return ensemble

    def _stacking(self, predictions: list) -> np.ndarray:
        """
        Stacking: usa meta-learner em cima das predições.

        Nota: Requer treinar meta-learner primeiro.
        """
        if self.meta_learner is None:
            logger.warning("Meta-learner não treinado, usando weighted average")
            return self._weighted_average(predictions)

        # Stack predições como features
        stacked_features = np.column_stack([pred for _, pred in predictions])

        # Meta-learner predição
        ensemble = self.meta_learner.predict_proba(stacked_features)[:, 1]

        return ensemble

    def train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sequence_length: int = 20
    ):
        """
        Treina meta-learner para stacking.

        Args:
            X_train: Features de treinamento
            y_train: Labels de treinamento
            sequence_length: Comprimento da sequência
        """
        from sklearn.linear_model import LogisticRegression

        logger.info("Treinando meta-learner para stacking...")

        # Gerar predições dos modelos base
        predictions = []

        if self.dl_model is not None:
            dl_pred = self.dl_model.predict(X_train, sequence_length=sequence_length)
            predictions.append(dl_pred)

        if self.tree_model is not None:
            X_tree = X_train[sequence_length:]
            if self.tree_type == 'xgboost':
                tree_pred = self.tree_model.predict_proba(X_tree)[:, 1]
            else:
                tree_pred = self.tree_model.predict_proba(X_tree)[:, 1]
            predictions.append(tree_pred)

        # Stack features
        stacked_features = np.column_stack(predictions)

        # Ajustar labels (remover sequências)
        y_train_adj = y_train[sequence_length:]

        # Treinar meta-learner
        self.meta_learner = LogisticRegression(random_state=42)
        self.meta_learner.fit(stacked_features, y_train_adj)

        logger.info("✓ Meta-learner treinado")

    def save(self, filepath: str):
        """
        Salva configuração do ensemble.
        """
        config = {
            'tree_type': self.tree_type,
            'ensemble_method': self.ensemble_method,
            'dl_weight': self.dl_weight,
            'tree_weight': self.tree_weight,
            'meta_learner': self.meta_learner
        }

        with open(filepath, 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"✓ Ensemble config salvo em {filepath}")

    def load(self, filepath: str):
        """
        Carrega configuração do ensemble.
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)

        self.tree_type = config['tree_type']
        self.ensemble_method = config['ensemble_method']
        self.dl_weight = config['dl_weight']
        self.tree_weight = config['tree_weight']
        self.meta_learner = config.get('meta_learner')

        logger.info(f"✓ Ensemble config carregado de {filepath}")


def compare_models(
    X_test: np.ndarray,
    y_test: np.ndarray,
    dl_pred: np.ndarray,
    tree_pred: np.ndarray,
    ensemble_pred: np.ndarray,
    sequence_length: int = 20
) -> pd.DataFrame:
    """
    Compara performance de DL, Tree e Ensemble.

    Returns:
        DataFrame com métricas comparativas
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    # Ajustar y_test (remover sequências)
    y_test_adj = y_test[sequence_length:]

    results = []

    # DL
    if dl_pred is not None:
        dl_pred_binary = (dl_pred >= 0.5).astype(int)
        results.append({
            'Model': 'Deep Learning',
            'ROC AUC': roc_auc_score(y_test_adj, dl_pred),
            'Accuracy': accuracy_score(y_test_adj, dl_pred_binary),
            'F1 Score': f1_score(y_test_adj, dl_pred_binary)
        })

    # Tree
    if tree_pred is not None:
        tree_pred_binary = (tree_pred >= 0.5).astype(int)
        results.append({
            'Model': 'Tree (XGB/LGB)',
            'ROC AUC': roc_auc_score(y_test_adj, tree_pred),
            'Accuracy': accuracy_score(y_test_adj, tree_pred_binary),
            'F1 Score': f1_score(y_test_adj, tree_pred_binary)
        })

    # Ensemble
    if ensemble_pred is not None:
        ensemble_pred_binary = (ensemble_pred >= 0.5).astype(int)
        results.append({
            'Model': 'Ensemble',
            'ROC AUC': roc_auc_score(y_test_adj, ensemble_pred),
            'Accuracy': accuracy_score(y_test_adj, ensemble_pred_binary),
            'F1 Score': f1_score(y_test_adj, ensemble_pred_binary)
        })

    df = pd.DataFrame(results)

    return df

"""
Ensemble Avançado: Stacking com Meta-Learner

Combina DL + LightGBM + XGBoost usando:
- Weighted Average
- Stacking (LogisticRegression ou XGBoost como meta-learner)
- Max Confidence
"""

import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


class StackingEnsemble:
    """
    Ensemble com Stacking Meta-Learner.
    
    Level 0: Base models (DL, LightGBM, XGBoost)
    Level 1: Meta-learner treina nas predições dos base models
    """
    
    def __init__(
        self,
        meta_learner_type='logistic_regression',
        cv_folds=5
    ):
        """
        Args:
            meta_learner_type: 'logistic_regression', 'xgboost', 'random_forest'
            cv_folds: Número de folds para cross-validation
        """
        self.meta_learner_type = meta_learner_type
        self.cv_folds = cv_folds
        self.meta_learner = None
        self.base_models = {}
    
    def add_base_model(self, name, model):
        """Adiciona modelo base ao ensemble"""
        self.base_models[name] = model
        logger.info(f"✓ Modelo base '{name}' adicionado ao ensemble")
    
    def train_meta_learner(self, X_train, y_train):
        """
        Treina meta-learner usando predições dos base models.
        
        Args:
            X_train: Features originais
            y_train: Labels
        """
        logger.info(f"Treinando meta-learner ({self.meta_learner_type})...")
        
        # Gerar predições dos base models
        base_predictions = []
        
        for name, model in self.base_models.items():
            logger.info(f"  Gerando predições de '{name}'...")
            
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X_train)[:, 1]
            elif hasattr(model, 'predict'):
                preds = model.predict(X_train)
                # Se retornou classes, converter para probabilidades
                if preds.dtype == int or preds.max() == 1:
                    preds = preds.astype(float)
            else:
                raise ValueError(f"Modelo '{name}' não tem método predict")
            
            base_predictions.append(preds)
        
        # Empilhar predições como features
        X_meta = np.column_stack(base_predictions)
        
        logger.info(f"  Meta features shape: {X_meta.shape}")
        
        # Treinar meta-learner
        if self.meta_learner_type == 'logistic_regression':
            self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.meta_learner_type == 'random_forest':
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Meta-learner '{self.meta_learner_type}' não reconhecido")
        
        self.meta_learner.fit(X_meta, y_train)
        
        # Score no treino
        train_score = self.meta_learner.score(X_meta, y_train)
        logger.info(f"✓ Meta-learner treinado - Accuracy: {train_score:.4f}")
    
    def predict(self, X):
        """
        Predição do ensemble.
        
        Args:
            X: Features
        
        Returns:
            Probabilidades de classe 1 (SHORT)
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner não foi treinado ainda!")
        
        # Gerar predições dos base models
        base_predictions = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
                if preds.dtype == int or preds.max() == 1:
                    preds = preds.astype(float)
            
            base_predictions.append(preds)
        
        # Empilhar
        X_meta = np.column_stack(base_predictions)
        
        # Predição do meta-learner
        if hasattr(self.meta_learner, 'predict_proba'):
            ensemble_pred = self.meta_learner.predict_proba(X_meta)[:, 1]
        else:
            ensemble_pred = self.meta_learner.predict(X_meta)
        
        return ensemble_pred
    
    def save(self, path):
        """Salva ensemble"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"✓ Ensemble salvo em: {path}")
    
    @classmethod
    def load(cls, path):
        """Carrega ensemble"""
        with open(path, 'rb') as f:
            ensemble = pickle.load(f)
        logger.info(f"✓ Ensemble carregado de: {path}")
        return ensemble


class WeightedEnsemble:
    """
    Ensemble simples com weighted average.
    Mais rápido que stacking mas menos flexível.
    """
    
    def __init__(self, weights=None):
        """
        Args:
            weights: Dict com pesos {'model_name': weight}
        """
        self.weights = weights or {}
        self.base_models = {}
    
    def add_base_model(self, name, model, weight=1.0):
        """Adiciona modelo base"""
        self.base_models[name] = model
        if name not in self.weights:
            self.weights[name] = weight
        logger.info(f"✓ '{name}' adicionado (weight={self.weights[name]:.2f})")
    
    def predict(self, X):
        """Predição weighted average"""
        # Normalizar pesos
        total_weight = sum(self.weights.values())
        norm_weights = {k: v/total_weight for k, v in self.weights.items()}
        
        ensemble_pred = np.zeros(len(X))
        
        for name, model in self.base_models.items():
            weight = norm_weights[name]
            
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            
            ensemble_pred += weight * preds
        
        return ensemble_pred


class MaxConfidenceEnsemble:
    """
    Ensemble que escolhe predição do modelo mais confiante.
    """
    
    def __init__(self):
        self.base_models = {}
    
    def add_base_model(self, name, model):
        """Adiciona modelo base"""
        self.base_models[name] = model
        logger.info(f"✓ '{name}' adicionado ao max confidence ensemble")
    
    def predict(self, X):
        """Escolhe predição mais confiante"""
        all_predictions = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            
            all_predictions.append(preds)
        
        all_predictions = np.array(all_predictions)  # (n_models, n_samples)
        
        # Calcular confiança (distância de 0.5)
        confidence = np.abs(all_predictions - 0.5)
        
        # Escolher modelo mais confiante para cada amostra
        most_confident = np.argmax(confidence, axis=0)
        
        # Pegar predição do modelo mais confiante
        ensemble_pred = np.array([
            all_predictions[model_idx, sample_idx]
            for sample_idx, model_idx in enumerate(most_confident)
        ])
        
        return ensemble_pred


def create_ensemble(config, models_dict):
    """
    Factory function para criar ensemble baseado em config.
    
    Args:
        config: Configuração YAML
        models_dict: Dict com modelos {'dl': model, 'lgb': model, 'xgb': model}
    
    Returns:
        Ensemble pronto
    """
    ensemble_config = config.get('ensemble', {})
    
    if not ensemble_config.get('enabled', False):
        logger.info("Ensemble desabilitado, usando apenas DL")
        return models_dict.get('deep_learning', models_dict.get('dl'))
    
    method = ensemble_config.get('method', 'weighted_average')
    
    if method == 'stacking':
        logger.info("🚀 Criando Stacking Ensemble")
        meta_learner = ensemble_config.get('meta_learner', 'logistic_regression')
        ensemble = StackingEnsemble(meta_learner_type=meta_learner)
        
        for name, model in models_dict.items():
            ensemble.add_base_model(name, model)
        
        return ensemble
    
    elif method == 'weighted_average':
        logger.info("🚀 Criando Weighted Ensemble")
        weights = ensemble_config.get('weights', {})
        ensemble = WeightedEnsemble(weights=weights)
        
        for name, model in models_dict.items():
            weight = weights.get(name, 1.0)
            ensemble.add_base_model(name, model, weight=weight)
        
        return ensemble
    
    elif method == 'max_confidence':
        logger.info("🚀 Criando Max Confidence Ensemble")
        ensemble = MaxConfidenceEnsemble()
        
        for name, model in models_dict.items():
            ensemble.add_base_model(name, model)
        
        return ensemble
    
    else:
        raise ValueError(f"Método de ensemble '{method}' não reconhecido")

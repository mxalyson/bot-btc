"""
Módulo para treinamento de modelos de Machine Learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
import joblib
from datetime import datetime

# ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM não instalado - apenas XGBoost disponível")

from core.utils import ensure_dir, save_model_metadata


class ModelTrainer:
    """
    Classe para treinar modelos de classificação para trading.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o trainer.

        Args:
            config: Dicionário de configuração
        """
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.dataset_config = config['dataset']

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_mapping = {'LONG': 0, 'SHORT': 1, 'NONE': 2}
        self.inverse_label_mapping = {0: 'LONG', 1: 'SHORT', 2: 'NONE'}

    def prepare_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara features para treinamento.

        Args:
            df: DataFrame com features e labels
            fit_scaler: Se True, ajusta o scaler. Se False, usa scaler já ajustado.

        Returns:
            Tupla (DataFrame com features normalizadas, lista de nomes de features)
        """
        # Identificar features (excluir colunas especiais)
        exclude_cols = self.dataset_config.get('exclude_features', [])
        all_cols = df.columns.tolist()
        feature_cols = [col for col in all_cols if col not in exclude_cols]

        # Garantir que temos as mesmas features no train/test
        if self.feature_names is not None:
            # Usar features do treino
            missing = set(self.feature_names) - set(feature_cols)
            if missing:
                raise ValueError(f"Features faltando no dataset: {missing}")
            feature_cols = self.feature_names
        else:
            # Primeira vez - salvar feature names
            self.feature_names = feature_cols

        # Extrair features
        X = df[feature_cols].copy()

        # Substituir inf por NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Remover linhas com NaN se configurado
        if self.dataset_config.get('drop_na', True):
            before = len(X)
            X = X.dropna()
            after = len(X)
            if before != after:
                logger.warning(f"Removidas {before - after} linhas com NaN")

        # Normalização
        if self.training_config.get('normalize_features', True):
            scaler_type = self.training_config.get('scaler_type', 'standard')

            if fit_scaler:
                # Criar e ajustar scaler
                if scaler_type == 'standard':
                    self.scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    self.scaler = MinMaxScaler()
                elif scaler_type == 'robust':
                    self.scaler = RobustScaler()
                else:
                    raise ValueError(f"Scaler desconhecido: {scaler_type}")

                X_scaled = self.scaler.fit_transform(X)
                logger.info(f"✓ Scaler ({scaler_type}) ajustado e aplicado")
            else:
                # Usar scaler existente
                if self.scaler is None:
                    raise ValueError("Scaler não foi ajustado ainda")
                X_scaled = self.scaler.transform(X)
                logger.info(f"✓ Scaler aplicado")

            X = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        logger.info(f"✓ Features preparadas: {X.shape}")

        return X, feature_cols

    def prepare_labels(self, df: pd.DataFrame, indices: pd.Index) -> np.ndarray:
        """
        Prepara labels para treinamento.

        Args:
            df: DataFrame com coluna 'target_class'
            indices: Índices para manter (mesmos que X após dropna)

        Returns:
            Array numpy com labels codificadas
        """
        if 'target_class' not in df.columns:
            raise ValueError("Coluna 'target_class' não encontrada")

        # Garantir alinhamento exato com índices de X
        # Reindex garante mesma ordem e quantidade de amostras
        y = df.reindex(indices)['target_class']

        # Verificar se há NaNs (índices que não existem no df)
        if y.isna().any():
            raise ValueError(f"Alguns índices de X não foram encontrados no DataFrame de labels")

        # Codificar labels
        y_encoded = y.map(self.label_mapping)

        # Verificar se há labels desconhecidos
        if y_encoded.isna().any():
            unknown = y[y_encoded.isna()].unique()
            raise ValueError(f"Labels desconhecidos encontrados: {unknown}")

        result = y_encoded.values
        logger.debug(f"prepare_labels: retornando array com shape {result.shape}")
        return result

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Treina o modelo.

        Args:
            df_train: DataFrame de treino
            df_val: DataFrame de validação (opcional)

        Returns:
            Dicionário com métricas de treinamento
        """
        logger.info("=" * 80)
        logger.info("INICIANDO TREINAMENTO DO MODELO".center(80))
        logger.info("=" * 80)

        # Preparar dados de treino
        X_train, feature_names = self.prepare_features(df_train, fit_scaler=True)
        y_train = self.prepare_labels(df_train, X_train.index)

        logger.info(f"Dataset de treino: {X_train.shape}")
        logger.info(f"y_train shape após prepare_labels: {y_train.shape}, dtype: {y_train.dtype}")
        logger.info(f"Distribuição de labels:")
        for label, code in self.label_mapping.items():
            count = (y_train == code).sum()
            pct = 100 * count / len(y_train)
            logger.info(f"  {label}: {count} ({pct:.2f}%)")

        # Preparar dados de validação (se fornecido)
        eval_set = None
        if df_val is not None:
            X_val, _ = self.prepare_features(df_val, fit_scaler=False)
            y_val = self.prepare_labels(df_val, X_val.index)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            logger.info(f"Dataset de validação: {X_val.shape}")

        # Treinar modelo
        model_type = self.model_config.get('type', 'xgboost')

        if model_type == 'xgboost':
            self.model = self._train_xgboost(X_train, y_train, eval_set)
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM não está instalado")
            self.model = self._train_lightgbm(X_train, y_train, eval_set)
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

        # Avaliar no treino
        logger.info("\n" + "=" * 80)
        logger.info("AVALIAÇÃO NO CONJUNTO DE TREINO".center(80))
        logger.info("=" * 80)
        train_metrics = self.evaluate(X_train, y_train)

        # Avaliar na validação (se disponível)
        val_metrics = None
        if df_val is not None:
            logger.info("\n" + "=" * 80)
            logger.info("AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO".center(80))
            logger.info("=" * 80)
            val_metrics = self.evaluate(X_val, y_val)

        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': self.get_feature_importance()
        }

        return results

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        eval_set: Optional[List] = None
    ) -> xgb.XGBClassifier:
        """
        Treina modelo XGBoost.
        """
        logger.info("Treinando XGBoost...")

        # Parâmetros do config
        params = self.model_config.get('xgboost', {}).copy()

        # Remover parâmetros que não são do XGBoost
        early_stopping = self.model_config.get('early_stopping_rounds', 50)

        # Criar modelo
        model = xgb.XGBClassifier(**params)

        # Treinar
        if eval_set is not None:
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
        else:
            model.fit(X_train, y_train, verbose=True)

        logger.info(f"✓ XGBoost treinado")

        return model

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        eval_set: Optional[List] = None
    ) -> lgb.LGBMClassifier:
        """
        Treina modelo LightGBM.
        """
        logger.info("Treinando LightGBM...")

        # Parâmetros do config
        params = self.model_config.get('lightgbm', {}).copy()

        # Criar modelo
        model = lgb.LGBMClassifier(**params)

        # Treinar
        callbacks = []
        if eval_set is not None:
            early_stopping = self.model_config.get('early_stopping_rounds', 50)
            callbacks.append(lgb.early_stopping(early_stopping))

            eval_names = ['train', 'valid']
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_names=eval_names,
                callbacks=callbacks
            )
        else:
            model.fit(X_train, y_train)

        logger.info(f"✓ LightGBM treinado")

        return model

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Avalia o modelo.

        Args:
            X: Features
            y: Labels verdadeiros

        Returns:
            Dicionário com métricas
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")

        logger.debug(f"evaluate: X shape = {X.shape}, y shape INICIAL = {y.shape}, y dtype = {y.dtype}")

        # Predições
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        logger.debug(f"evaluate: y_pred shape inicial = {y_pred.shape}, dtype = {y_pred.dtype}")

        # Garantir que y_pred é 1D array de integers
        # Se y_pred for 2D (probabilidades por classe), pegar argmax
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            logger.warning(f"y_pred é 2D com shape {y_pred.shape}, usando argmax para obter classe predita")
            y_pred = np.argmax(y_pred, axis=1)
        elif len(y_pred.shape) > 1:
            # Se for 2D mas com apenas 1 coluna, fazer flatten
            logger.warning(f"y_pred é 2D com shape {y_pred.shape}, aplicando flatten")
            y_pred = y_pred.flatten()

        y_pred = y_pred.astype(int)

        # Garantir que y é 1D array
        if len(y.shape) > 1:
            logger.warning(f"y é 2D com shape {y.shape}, aplicando flatten")
            y = y.flatten()

        logger.debug(f"evaluate: Após ajustes - X: {X.shape}, y: {y.shape}, y_pred: {y_pred.shape}")

        # Métricas gerais
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        precision_macro = precision_score(y, y_pred, average='macro')
        recall_macro = recall_score(y, y_pred, average='macro')

        logger.info(f"Accuracy:        {accuracy:.4f}")
        logger.info(f"F1 (macro):      {f1_macro:.4f}")
        logger.info(f"F1 (weighted):   {f1_weighted:.4f}")
        logger.info(f"Precision:       {precision_macro:.4f}")
        logger.info(f"Recall:          {recall_macro:.4f}")

        # Matriz de confusão
        # Detectar classes únicas presentes nos dados
        unique_classes = sorted(np.unique(np.concatenate([y, y_pred])))
        class_names = [self.inverse_label_mapping.get(c, f'Class_{c}') for c in unique_classes]

        cm = confusion_matrix(y, y_pred, labels=unique_classes)
        logger.info(f"\nMatriz de Confusão:")

        # Header dinâmico
        header = f"{'':>12}"
        for name in class_names:
            header += f" {name:>10}"
        logger.info(header)

        # Linhas da matriz
        for i, label in enumerate(class_names):
            row = f"{label:>12}"
            for j in range(len(class_names)):
                row += f" {cm[i][j]:>10}"
            logger.info(row)

        # Relatório de classificação
        logger.info(f"\nRelatório de Classificação:")
        report = classification_report(
            y, y_pred,
            labels=unique_classes,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        logger.info(f"\n{report}")

        # ROC AUC (multiclass)
        try:
            roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
            logger.info(f"ROC AUC (macro): {roc_auc:.4f}")
        except Exception as e:
            logger.warning(f"Não foi possível calcular ROC AUC: {e}")
            roc_auc = None

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc
        }

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna importância das features.

        Args:
            top_n: Número de features mais importantes

        Returns:
            DataFrame com feature importance
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")

        # Extrair importância
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logger.warning("Modelo não suporta feature importance")
            return pd.DataFrame()

        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Top N
        top_features = importance_df.head(top_n)

        logger.info(f"\nTop {top_n} Features Mais Importantes:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']:30s} {row['importance']:.6f}")

        return importance_df

    def save_model(self, model_name: Optional[str] = None) -> Path:
        """
        Salva o modelo treinado e metadados.

        Args:
            model_name: Nome do modelo (se None, usa timestamp)

        Returns:
            Path do arquivo salvo
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")

        # Criar diretório
        output_dir = Path(self.model_config.get('output_dir', 'models'))
        ensure_dir(output_dir)

        # Nome do arquivo
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"model_{timestamp}"

        model_path = output_dir / f"{model_name}.pkl"

        # Salvar modelo completo (modelo + scaler + features)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping,
            'inverse_label_mapping': self.inverse_label_mapping
        }

        joblib.dump(model_data, model_path)
        logger.info(f"✓ Modelo salvo em: {model_path}")

        # Salvar metadados
        metadata = {
            'model_type': self.model_config.get('type'),
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'scaler_type': self.training_config.get('scaler_type'),
            'config': self.config
        }

        save_model_metadata(model_path, metadata)

        return model_path

    def load_model(self, model_path: Path) -> None:
        """
        Carrega modelo salvo.

        Args:
            model_path: Caminho do modelo
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.label_mapping = model_data['label_mapping']
        self.inverse_label_mapping = model_data['inverse_label_mapping']

        logger.info(f"✓ Modelo carregado de: {model_path}")


if __name__ == "__main__":
    # Teste do módulo
    from core.utils import load_config, setup_logging
    from core.data_loader import BybitDataLoader
    from core.feature_engineering import FeatureEngineer
    from core.labeling import TradingLabeler

    config = load_config()
    setup_logging(config)

    # Pipeline completo de teste
    loader = BybitDataLoader(config)
    df = loader.download_ohlcv('BTCUSDT', '5m', lookback_days=60)

    engineer = FeatureEngineer(config)
    df = engineer.create_all_features(df)

    labeler = TradingLabeler(config)
    df = labeler.create_labels(df)

    # Dividir dados
    df_train, df_val, df_test = labeler.split_temporal(df)

    # Treinar
    trainer = ModelTrainer(config)
    results = trainer.train(df_train, df_val)

    # Salvar
    model_path = trainer.save_model('test_model')
    print(f"\nModelo salvo em: {model_path}")

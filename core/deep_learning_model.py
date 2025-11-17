"""
Arquitetura Deep Learning para Scalping de Criptomoedas
Modelo Híbrido: CNN + Bidirectional LSTM + Multi-Head Attention

Inspirado em:
- DeepLOB (order book deep learning)
- Temporal Fusion Transformer
- Research mostra: arquiteturas híbridas superam modelos isolados
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple, Optional
import pickle
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow não disponível. Install com: pip install tensorflow")


class AttentionLayer(layers.Layer):
    """
    Multi-Head Attention customizada para time series.
    Permite o modelo focar nos momentos mais importantes da sequência.
    """

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        # Compute attention scores
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)

        # Apply attention weights
        ait = tf.expand_dims(ait, -1)
        weighted_input = x * ait

        # Sum over timesteps
        output = tf.reduce_sum(weighted_input, axis=1)

        return output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class DeepScalpingModel:
    """
    Modelo híbrido CNN+LSTM+Attention para scalping de criptomoedas.

    Arquitetura:
    1. Conv1D: Captura padrões locais (3-5 candles)
    2. Bidirectional LSTM: Dependências temporais longas
    3. Attention: Foca nos momentos críticos
    4. Dense + Dropout: Classificação final
    """

    def __init__(self, config: dict):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow é necessário. Install: pip install tensorflow")

        self.config = config
        self.model = None
        self.scaler = None
        self.history = None

    def build_model(
        self,
        sequence_length: int,
        n_features: int,
        config: Optional[dict] = None
    ) -> Model:
        """
        Constrói a arquitetura do modelo.
        """
        if config is None:
            config = self.config.get('model', {}).get('deep_learning', {})

        # Hiperparâmetros
        conv_filters = config.get('conv_filters', 64)
        conv_kernel = config.get('conv_kernel', 3)
        lstm_units = config.get('lstm_units', 128)
        attention_units = config.get('attention_units', 64)
        dense_units = config.get('dense_units', 64)
        dropout_rate = config.get('dropout_rate', 0.3)
        l2_reg = config.get('l2_reg', 0.001)

        logger.info(f"Construindo modelo Deep Learning...")
        logger.info(f"  Sequence length: {sequence_length}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Conv filters: {conv_filters}")
        logger.info(f"  LSTM units: {lstm_units}")
        logger.info(f"  Attention units: {attention_units}")

        # Input
        inputs = layers.Input(shape=(sequence_length, n_features), name='input')

        # === BLOCO 1: CNN para padrões locais ===
        x = layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel,
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='conv1d_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout_conv1')(x)

        # Segundo bloco CNN
        x = layers.Conv1D(
            filters=conv_filters // 2,
            kernel_size=conv_kernel,
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='conv1d_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout_conv2')(x)

        # === BLOCO 2: Bidirectional LSTM ===
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                recurrent_regularizer=keras.regularizers.l2(l2_reg),
                name='lstm_1'
            ),
            name='bidirectional_lstm'
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(dropout_rate, name='dropout_lstm')(x)

        # === BLOCO 3: Attention ===
        x = AttentionLayer(attention_units, name='attention')(x)

        # === BLOCO 4: Dense Layers ===
        x = layers.Dense(
            dense_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_4')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense')(x)

        # Output (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Criar modelo
        model = Model(inputs=inputs, outputs=outputs, name='DeepScalpingModel')

        # Compilar
        optimizer = keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.001)
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        self.model = model

        logger.info(f"✓ Modelo construído com sucesso")
        logger.info(f"  Total parameters: {model.count_params():,}")

        return model

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converte dados tabulares em sequências para LSTM.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            sequence_length: Comprimento da sequência

        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,)
        """
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 64
    ) -> dict:
        """
        Treina o modelo.
        """
        logger.info("Iniciando treinamento Deep Learning...")

        # Criar sequências
        logger.info(f"Criando sequências (length={sequence_length})...")
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, sequence_length)

        logger.info(f"  Train sequences: {X_train_seq.shape}")
        logger.info(f"  Val sequences: {X_val_seq.shape}")

        # Construir modelo se não existir
        if self.model is None:
            self.build_model(
                sequence_length=sequence_length,
                n_features=X_train.shape[1]
            )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=15,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=5,
                mode='max',
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Class weights para balanceamento
        class_counts = np.bincount(y_train_seq.astype(int))
        total = len(y_train_seq)
        class_weight = {
            0: total / (2 * class_counts[0]),  # LONG
            1: total / (2 * class_counts[1])   # SHORT
        }

        logger.info(f"Class weights: {class_weight}")

        # Treinar
        logger.info(f"Treinando por até {epochs} epochs...")
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        # Resultados
        train_metrics = self.model.evaluate(X_train_seq, y_train_seq, verbose=0)
        val_metrics = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)

        results = {
            'train_loss': train_metrics[0],
            'train_accuracy': train_metrics[1],
            'train_auc': train_metrics[2],
            'val_loss': val_metrics[0],
            'val_accuracy': val_metrics[1],
            'val_auc': val_metrics[2],
            'epochs_trained': len(self.history.history['loss'])
        }

        logger.info("=" * 80)
        logger.info("RESULTADOS DO TREINAMENTO")
        logger.info("=" * 80)
        logger.info(f"Train - Loss: {results['train_loss']:.4f}, Accuracy: {results['train_accuracy']:.4f}, AUC: {results['train_auc']:.4f}")
        logger.info(f"Val   - Loss: {results['val_loss']:.4f}, Accuracy: {results['val_accuracy']:.4f}, AUC: {results['val_auc']:.4f}")
        logger.info(f"Epochs: {results['epochs_trained']}")
        logger.info("=" * 80)

        return results

    def predict(self, X: np.ndarray, sequence_length: int = 20) -> np.ndarray:
        """
        Faz predições.
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")

        # Criar sequências
        X_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])

        X_seq = np.array(X_seq)

        # Prever
        y_proba = self.model.predict(X_seq, verbose=0)

        return y_proba.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 20) -> dict:
        """
        Avalia o modelo.
        """
        X_seq, y_seq = self.create_sequences(X, y, sequence_length)

        metrics = self.model.evaluate(X_seq, y_seq, verbose=0)

        results = {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'auc': metrics[2],
            'precision': metrics[3],
            'recall': metrics[4]
        }

        return results

    def save(self, filepath: str):
        """
        Salva o modelo.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Salvar modelo Keras
        self.model.save(str(filepath) + '.keras')

        # Salvar scaler separadamente
        if self.scaler is not None:
            with open(str(filepath) + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        logger.info(f"✓ Modelo salvo em {filepath}")

    def load(self, filepath: str):
        """
        Carrega o modelo.
        """
        filepath = Path(filepath)

        # Carregar modelo Keras
        self.model = keras.models.load_model(
            str(filepath) + '.keras',
            custom_objects={'AttentionLayer': AttentionLayer}
        )

        # Carregar scaler
        scaler_path = str(filepath) + '_scaler.pkl'
        if Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        logger.info(f"✓ Modelo carregado de {filepath}")

    def summary(self):
        """
        Mostra sumário do modelo.
        """
        if self.model:
            self.model.summary()
        else:
            logger.warning("Modelo não construído ainda")

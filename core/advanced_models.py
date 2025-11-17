"""
Arquiteturas Avançadas de Deep Learning para Trading

Inclui:
- Transformer Architecture
- Multi-Scale CNN
- Residual Connections
- Multi-Head Attention melhorado
- Focal Loss
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from loguru import logger


# ===== FOCAL LOSS =====

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss para focar em exemplos difíceis.
    
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        gamma: Fator de modulação (default 2.0)
        alpha: Balanceamento de classes (default 0.25)
    """
    def loss_fn(y_true, y_pred):
        # Clip para estabilidade numérica
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilidade do ground truth
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Focal weight
        focal_weight = alpha * tf.pow(1 - pt, gamma)
        
        # Binary crossentropy
        bce = -tf.math.log(pt)
        
        # Focal loss
        loss = focal_weight * bce
        
        return tf.reduce_mean(loss)
    
    return loss_fn


# ===== POSITIONAL ENCODING (para Transformer) =====

class PositionalEncoding(layers.Layer):
    """
    Positional Encoding sinusoidal para Transformers.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, max_len=100, d_model=128):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Pre-compute positional encodings
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


# ===== TRANSFORMER BLOCK =====

class TransformerBlock(layers.Layer):
    """
    Bloco Transformer completo:
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections
    """
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-Head Attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        
        # Feed-Forward Network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Self-Attention + Residual
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection
        
        # Feed-Forward + Residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2


# ===== TRANSFORMER MODEL =====

class TransformerModel:
    """
    Transformer completo para Trading.
    
    Architecture:
    - Input Projection
    - Positional Encoding
    - N x Transformer Blocks
    - Global Average Pooling
    - Dense Classifier
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def build(self, sequence_length, n_features):
        """Constrói arquitetura Transformer"""
        
        transformer_config = self.config['model']['transformer']
        
        d_model = transformer_config['d_model']
        num_heads = transformer_config['num_heads']
        num_layers = transformer_config['num_layers']
        ff_dim = transformer_config['ff_dim']
        dropout = transformer_config['dropout']
        
        logger.info(f"Construindo Transformer: d_model={d_model}, heads={num_heads}, layers={num_layers}")
        
        # Input
        inputs = layers.Input(shape=(sequence_length, n_features))
        
        # Input Projection (transformar n_features -> d_model)
        x = layers.Dense(d_model)(inputs)
        
        # Positional Encoding
        x = PositionalEncoding(max_len=sequence_length, d_model=d_model)(x)
        
        # N x Transformer Blocks
        for _ in range(num_layers):
            x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)
        
        # Global Average Pooling (reduz sequência para vetor)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Contar parâmetros
        total_params = self.model.count_params()
        logger.info(f"✓ Transformer construído: {total_params:,} parâmetros")
        
        return self.model


# ===== HYBRID MODEL (Multi-Scale CNN + LSTM + Attention) =====

class HybridModel:
    """
    Modelo Híbrido Melhorado:
    - Multi-Scale CNN (3 escalas)
    - Bidirectional LSTM com Residual Connections
    - Multi-Head Attention
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def build(self, sequence_length, n_features):
        """Constrói arquitetura híbrida melhorada"""
        
        cnn_lstm_config = self.config['model']['deep_learning']
        
        conv_filters = cnn_lstm_config['conv_filters']
        conv_kernels = cnn_lstm_config['conv_kernels']
        lstm_units = cnn_lstm_config['lstm_units']
        attention_units = cnn_lstm_config['attention_units']
        dropout = cnn_lstm_config['dropout_rate']
        use_residual = cnn_lstm_config.get('use_residual', True)
        
        logger.info(f"Construindo Hybrid Model (Multi-Scale CNN + LSTM + Attention)")
        
        # Input
        inputs = layers.Input(shape=(sequence_length, n_features))
        
        # === MULTI-SCALE CNN ===
        conv_outputs = []
        for filters, kernel in zip(conv_filters, conv_kernels):
            conv = layers.Conv1D(filters, kernel, activation='relu', padding='same')(inputs)
            conv = layers.BatchNormalization()(conv)
            conv_outputs.append(conv)
        
        # Concatenar outputs das 3 escalas
        x = layers.Concatenate()(conv_outputs) if len(conv_outputs) > 1 else conv_outputs[0]
        x = layers.Dropout(dropout)(x)
        
        logger.info(f"  Multi-Scale CNN: {len(conv_kernels)} escalas {conv_kernels}")
        
        # === BIDIRECTIONAL LSTM com Residual ===
        lstm_out = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True)
        )(x)
        lstm_out = layers.Dropout(dropout)(lstm_out)
        
        # Residual connection (se dimensões compatíveis)
        if use_residual:
            # Projetar x para mesma dimensão do LSTM output
            x_projected = layers.Conv1D(lstm_units * 2, 1)(x)  # *2 porque Bidirectional
            x = layers.Add()([x_projected, lstm_out])  # Residual
            logger.info("  ✓ Residual connections habilitados")
        else:
            x = lstm_out
        
        x = layers.LayerNormalization()(x)
        
        # === MULTI-HEAD ATTENTION ===
        attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=attention_units
        )(x, x)
        
        x = layers.Add()([x, attention])  # Residual
        x = layers.LayerNormalization()(x)
        
        logger.info(f"  Multi-Head Attention: 8 heads, key_dim={attention_units}")
        
        # === POOLING ===
        x = layers.GlobalAveragePooling1D()(x)
        
        # === DENSE CLASSIFIER ===
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        total_params = self.model.count_params()
        logger.info(f"✓ Hybrid Model construído: {total_params:,} parâmetros")
        
        return self.model


# ===== FACTORY FUNCTION =====

def build_advanced_model(config, sequence_length, n_features):
    """
    Constrói modelo baseado em configuração.
    
    Args:
        config: Configuração YAML
        sequence_length: Comprimento da sequência
        n_features: Número de features
    
    Returns:
        Modelo Keras compilado
    """
    architecture = config['model'].get('architecture', 'cnn_lstm')
    
    if architecture == 'transformer':
        logger.info("🚀 Usando arquitetura TRANSFORMER")
        model_builder = TransformerModel(config)
    elif architecture == 'hybrid' or architecture == 'cnn_lstm':
        logger.info("🚀 Usando arquitetura HYBRID (Multi-Scale CNN + LSTM + Attention)")
        model_builder = HybridModel(config)
    else:
        raise ValueError(f"Arquitetura '{architecture}' não reconhecida")
    
    # Construir modelo
    model = model_builder.build(sequence_length, n_features)
    
    # Compilar
    training_config = config['training']
    
    # Loss function
    loss_type = training_config.get('loss', 'binary_crossentropy')
    if loss_type == 'focal_loss':
        gamma = training_config.get('focal_loss_gamma', 2.0)
        alpha = training_config.get('focal_loss_alpha', 0.25)
        loss = focal_loss(gamma=gamma, alpha=alpha)
        logger.info(f"✓ Usando Focal Loss (gamma={gamma}, alpha={alpha})")
    else:
        loss = 'binary_crossentropy'
        logger.info("✓ Usando Binary Crossentropy")
    
    # Optimizer
    lr = training_config.get('learning_rate', 0.001)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    # Métricas
    metrics = [
        keras.metrics.AUC(name='auc'),
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logger.info("✓ Modelo compilado e pronto para treinamento")
    
    return model

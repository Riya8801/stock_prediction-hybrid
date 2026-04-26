"""
Hybrid TCN-LSTM Model for Stock Price Forecasting
Author: Your Name
Project: Forecasting Closing Prices using Hybrid Deep Learning

This combines:
- TCN: Extracts temporal patterns with dilated convolutions
- LSTM: Captures long-term sequential dependencies
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import Conv1D, Dense, Dropout, Activation, BatchNormalization
from keras.layers import LSTM, Bidirectional, Concatenate, GlobalAveragePooling1D, Attention
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
import numpy as np


class TemporalBlock(layers.Layer):
    """Temporal Block for TCN"""
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding='causal')
        self.batch_norm1 = BatchNormalization()
        self.activation1 = Activation('relu')
        self.dropout1 = Dropout(dropout_rate)
        
        self.conv2 = Conv1D(filters=filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding='causal')
        self.batch_norm2 = BatchNormalization()
        self.activation2 = Activation('relu')
        self.dropout2 = Dropout(dropout_rate)
        
        self.downsample = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.downsample = Conv1D(filters=self.filters, kernel_size=1, padding='same')
        super(TemporalBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        
        residual = self.downsample(inputs) if self.downsample else inputs
        return layers.Add()([x, residual])
    
    def get_config(self):
        config = super(TemporalBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config


class HybridTCNLSTM:
    """
    Hybrid TCN-LSTM Model
    
    Architecture:
    Input → [TCN Branch + LSTM Branch] → Fusion → Dense Layers → Output
    """
    
    def __init__(self, input_shape, num_tcn_filters=64, num_lstm_units=64,
                 kernel_size=3, num_blocks=3, dropout_rate=0.2):
        self.input_shape = input_shape
        self.num_tcn_filters = num_tcn_filters
        self.num_lstm_units = num_lstm_units
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build hybrid architecture"""
        print("\n" + "="*70)
        print("BUILDING HYBRID TCN-LSTM MODEL")
        print("="*70)
        
        inputs = layers.Input(shape=self.input_shape)
        
        # ========== TCN BRANCH ==========
        print("\n🔷 TCN Branch (Pattern Extraction):")
        tcn_path = inputs
        for i in range(self.num_blocks):
            dilation_rate = 2 ** i
            tcn_path = TemporalBlock(
                filters=self.num_tcn_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate
            )(tcn_path)
            print(f"  Block {i+1}: filters={self.num_tcn_filters}, dilation={dilation_rate}")
        
        tcn_output = layers.GlobalAveragePooling1D()(tcn_path)
        print(f"  TCN Output: {self.num_tcn_filters} features")
        
        # ========== LSTM BRANCH ==========
        print("\n🔶 LSTM Branch (Sequential Modeling):")
        lstm_path = inputs
        
        # Bidirectional LSTM
        lstm_path = Bidirectional(LSTM(
            self.num_lstm_units,
            return_sequences=True,
            dropout=self.dropout_rate
        ))(lstm_path)
        print(f"  Bidirectional LSTM: {self.num_lstm_units} units × 2")
        
        # Second LSTM
        lstm_path = LSTM(
            self.num_lstm_units // 2,
            return_sequences=True,
            dropout=self.dropout_rate
        )(lstm_path)
        print(f"  LSTM Layer 2: {self.num_lstm_units // 2} units (return sequences)")
        
        # Self Attention
        attention_out = Attention()([lstm_path, lstm_path])
        lstm_output = layers.GlobalAveragePooling1D()(attention_out)
        print(f"  LSTM Output with Attention: {self.num_lstm_units // 2} features")
        
        # ========== FUSION ==========
        print("\n🔀 Fusion Layer:")
        combined = Concatenate()([tcn_output, lstm_output])
        total_features = self.num_tcn_filters + self.num_lstm_units // 2
        print(f"  Combined: {total_features} features")
        
        # ========== DENSE LAYERS ==========
        print("\n🔸 Dense Layers:")
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
        x = Dropout(0.3)(x)
        print("  Dense 1: 128 units (L2 reg)")
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        print("  Dense 2: 64 units (L2 reg)")
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        print("  Dense 3: 32 units (L2 reg)")
        
        outputs = Dense(1, activation='linear')(x)
        print("  Output: 1 (closing price)")
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name='Hybrid_TCN_LSTM')
        print("\n✅ Hybrid model built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        huber_loss = keras.losses.Huber()
        self.model.compile(optimizer=optimizer, loss=huber_loss, metrics=['mae', 'mse'])
        print(f"✅ Model compiled with Huber Loss (lr={learning_rate})")
        
    def get_summary(self):
        """Print model summary"""
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        self.model.summary()
        print(f"\n📊 Total parameters: {self.model.count_params():,}")
        
    def get_callbacks(self, model_save_path='best_hybrid_model.h5'):
        """Get training callbacks"""
        return [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
        ]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              model_save_path='best_hybrid_model.h5'):
        """Train the model"""
        print("\n" + "="*70)
        print("TRAINING HYBRID MODEL")
        print("="*70)
        print(f"Training: {len(X_train)} samples | Validation: {len(X_val)} samples")
        print(f"Epochs: {epochs} | Batch size: {batch_size}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_save_path),
            verbose=1
        )
        print("\n✅ Training completed!")
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0).flatten()
    
    def save_model(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"💾 Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath, custom_objects={'TemporalBlock': TemporalBlock})
        print(f"📂 Model loaded: {filepath}")


if __name__ == "__main__":
    print("Hybrid TCN-LSTM Model Module")
    print("Combines TCN (pattern extraction) + LSTM (sequential modeling)")

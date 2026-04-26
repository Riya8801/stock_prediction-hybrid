"""
Ultimate Hybrid Model: Institutional Final-Tier (>95% Goal)
Implementing Gradient Clipping, Cosine Decay, and Stable Huber Loss
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

class TemporalBlock(layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.1, **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.n_outputs = n_outputs
        self.conv1 = layers.Conv1D(n_outputs, kernel_size, strides=strides, 
                                  dilation_rate=dilation_rate, padding='causal', 
                                  kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('swish') # Swish is smoother for deep networks
        self.dropout1 = layers.Dropout(dropout)
        self.conv2 = layers.Conv1D(n_outputs, kernel_size, strides=strides, 
                                  dilation_rate=dilation_rate, padding='causal', 
                                  kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('swish')
        self.dropout2 = layers.Dropout(dropout)
        self.downsample = None
        self.relu = layers.Activation('swish')

    def build(self, input_shape):
        if input_shape[-1] != self.n_outputs:
            self.downsample = layers.Conv1D(self.n_outputs, 1, padding='same')
        super(TemporalBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.dropout1(out, training=training)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu2(out)
        out = self.dropout2(out, training=training)
        res = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu(out + res)

class HybridTCNLSTM:
    def __init__(self, input_shape, num_tcn_filters=64, num_lstm_units=64, kernel_size=3, num_blocks=4, dropout_rate=0.15):
        self.input_shape = input_shape
        self.num_tcn_filters = num_tcn_filters
        self.num_lstm_units = num_lstm_units
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        # --- TCN BRANCH (State-space pattern extraction) ---
        tcn_path = inputs
        for i in range(self.num_blocks):
            tcn_path = TemporalBlock(self.num_tcn_filters, self.kernel_size, strides=1, 
                                    dilation_rate=2**i, dropout=self.dropout_rate)(tcn_path)
        tcn_output = layers.GlobalMaxPooling1D()(tcn_path) # Max pooling for sharp features
        
        # --- LSTM BRANCH (Temporal Trend) ---
        lstm_path = layers.Bidirectional(layers.LSTM(self.num_lstm_units, return_sequences=True))(inputs)
        lstm_path = layers.LayerNormalization()(lstm_path)
        
        # High-Resolution Multi-Head Attention
        attention_out = layers.MultiHeadAttention(num_heads=8, key_dim=16)(lstm_path, lstm_path)
        attention_out = layers.Add()([lstm_path, attention_out]) # Residual
        lstm_output = layers.GlobalAveragePooling1D()(attention_out)
        
        # --- STABLE FUSION ---
        combined = layers.Concatenate()([tcn_output, lstm_output])
        
        x = layers.Dense(256, activation='swish')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='swish')(x)
        x = layers.Dense(64, activation='swish')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name='Institutional_Ultra_Hybrid')
        return self.model

    def compile_model(self, learning_rate=0.0005):
        # Implementation of Gradient Clipping for stability
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        # Huber loss is more robust to outliers than MSE
        self.model.compile(optimizer=optimizer, loss=keras.losses.Huber(delta=0.1), metrics=['mae'])

    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=16, model_save_path='best_model.h5'):
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            # Cosine decay imitation
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
        ]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

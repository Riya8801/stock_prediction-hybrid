"""
Temporal Convolutional Network (TCN) Model
Author: Your Name
Project: Forecasting Closing Prices using TCN
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import Conv1D, Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np


class TemporalBlock(layers.Layer):
    """
    Temporal Block - the building block of TCN
    """
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # First convolutional layer
        self.conv1 = Conv1D(filters=filters, 
                           kernel_size=kernel_size,
                           dilation_rate=dilation_rate,
                           padding='causal',
                           activation='linear')
        self.batch_norm1 = BatchNormalization()
        self.activation1 = Activation('relu')
        self.dropout1 = Dropout(dropout_rate)
        
        # Second convolutional layer
        self.conv2 = Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           dilation_rate=dilation_rate,
                           padding='causal',
                           activation='linear')
        self.batch_norm2 = BatchNormalization()
        self.activation2 = Activation('relu')
        self.dropout2 = Dropout(dropout_rate)
        
        # Residual connection (1x1 conv to match dimensions if needed)
        self.downsample = None
        
    def build(self, input_shape):
        # If input channels != output channels, use 1x1 conv for residual
        if input_shape[-1] != self.filters:
            self.downsample = Conv1D(filters=self.filters, kernel_size=1, padding='same')
        super(TemporalBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # First convolution block
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        # Second convolution block
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        
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


class TCNModel:
    """
    Temporal Convolutional Network for Stock Price Prediction
    """
    
    def __init__(self, input_shape, num_filters=64, kernel_size=3, 
                 num_blocks=4, dropout_rate=0.2):
        """
        Initialize TCN model
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_filters: Number of filters in convolutional layers
            kernel_size: Size of convolutional kernel
            num_blocks: Number of temporal blocks
            dropout_rate: Dropout rate
        """
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """
        Build the TCN model architecture
        """
        print("\n" + "="*50)
        print("BUILDING TCN MODEL")
        print("="*50)
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial layer
        x = inputs
        
        # Stack temporal blocks with increasing dilation rates
        for i in range(self.num_blocks):
            dilation_rate = 2 ** i
            x = TemporalBlock(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate
            )(x)
            print(f"Block {i+1}: filters={self.num_filters}, dilation_rate={dilation_rate}")
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer (single value - closing price)
        outputs = layers.Dense(1, activation='linear')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        print("\nModel built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        print(f"\nCompiling model with learning_rate={learning_rate}...")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error
            metrics=['mae', 'mse']  # Mean Absolute Error, Mean Squared Error
        )
        
        print("Model compiled successfully!")
        
    def get_summary(self):
        """
        Print model summary
        """
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        self.model.summary()
        
        # Calculate total parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
    def get_callbacks(self, model_save_path='best_tcn_model.h5'):
        """
        Get training callbacks
        
        Args:
            model_save_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, model_save_path='best_tcn_model.h5'):
        """
        Train the model
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size
            model_save_path: Path to save best model
            
        Returns:
            Training history
        """
        print("\n" + "="*50)
        print("TRAINING TCN MODEL")
        print("="*50)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        callbacks = self.get_callbacks(model_save_path)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def save_model(self, filepath):
        """
        Save the model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
        """
        # Register custom objects
        custom_objects = {'TemporalBlock': TemporalBlock}
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("TCN Model Module")
    print("This module implements Temporal Convolutional Network for stock price prediction")
    
    # Example model creation
    input_shape = (60, 5)  # 60 time steps, 5 features
    tcn = TCNModel(input_shape=input_shape, num_filters=64, kernel_size=3, num_blocks=4)
    tcn.build_model()
    tcn.compile_model()
    tcn.get_summary()

"""
Main Training Script for Stock Price Forecasting using TCN
Author: Your Name
Project: Forecasting Closing Prices using TCN

This script orchestrates the entire pipeline:
1. Data loading and preprocessing
2. Model building and training
3. Evaluation and visualization
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import custom modules
from data_preprocessing import StockDataPreprocessor
from tcn_model import TCNModel
from model_evaluation import ModelEvaluator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ================= PATH CONFIGURATION =================
BASE_DIR = os.getcwd()   # stock_tcn_project
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# =====================================================

class StockPriceForecaster:
    """
    Main class to orchestrate the stock price forecasting pipeline
    """
    
    def __init__(self, data_path, company_name=None):
        """
        Initialize the forecaster
        
        Args:
            data_path: Path to the CSV data file
            company_name: Name of the company to analyze (None = first company)
        """
        self.data_path = data_path
        self.company_name = company_name
        self.preprocessor = None
        self.tcn_model = None
        self.evaluator = None
        self.history = None
        
    def run_pipeline(self, sequence_length=60, train_ratio=0.8, 
                    num_filters=64, kernel_size=3, num_blocks=4,
                    epochs=100, batch_size=32, learning_rate=0.001):
        """
        Run the complete forecasting pipeline
        
        Args:
            sequence_length: Number of past days to consider
            train_ratio: Ratio of data to use for training
            num_filters: Number of filters in TCN
            kernel_size: Kernel size for convolution
            num_blocks: Number of temporal blocks
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        print("\n" + "="*70)
        print(" STOCK PRICE FORECASTING USING TEMPORAL CONVOLUTIONAL NETWORK (TCN) ")
        print("="*70)
        
        # ============================================================
        # STEP 1: DATA PREPROCESSING
        # ============================================================
        print("\n[STEP 1] DATA PREPROCESSING")
        print("-" * 70)
        
        self.preprocessor = StockDataPreprocessor(self.data_path)
        
        # Load data
        data = self.preprocessor.load_data()
        
        # Explore data
        self.preprocessor.explore_data()
        
        # Handle missing values
        self.preprocessor.handle_missing_values()
        
        # Visualize data
        self.preprocessor.visualize_data(self.company_name)
        
        # Prepare features
        feature_data = self.preprocessor.prepare_features(self.company_name)
        
        # Scale data
        scaled_data = self.preprocessor.scale_data()
        
        # Create sequences
        X, y = self.preprocessor.create_sequences(sequence_length=sequence_length)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            X, y, train_ratio=train_ratio
        )
        
        print("\n[STEP 1 COMPLETED] Data preprocessing finished successfully!")
        
        # ============================================================
        # STEP 2: MODEL BUILDING AND TRAINING
        # ============================================================
        print("\n[STEP 2] MODEL BUILDING AND TRAINING")
        print("-" * 70)
        
        # Build TCN model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.tcn_model = TCNModel(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout_rate=0.2
        )
        
        self.tcn_model.build_model()
        self.tcn_model.compile_model(learning_rate=learning_rate)
        self.tcn_model.get_summary()
        
        # Train model
        self.history = self.tcn_model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            model_save_path=os.path.join(MODELS_DIR, 'best_tcn_model.h5')

        )
        
        print("\n[STEP 2 COMPLETED] Model training finished successfully!")
        
        # ============================================================
        # STEP 3: EVALUATION AND VISUALIZATION
        # ============================================================
        print("\n[STEP 3] MODEL EVALUATION AND VISUALIZATION")
        print("-" * 70)
        
        # Make predictions
        y_train_pred_scaled = self.tcn_model.predict(X_train)
        y_test_pred_scaled = self.tcn_model.predict(X_test)
        
        # Inverse transform predictions to original scale
        y_train_true = self.preprocessor.inverse_transform_predictions(y_train)
        y_train_pred = self.preprocessor.inverse_transform_predictions(y_train_pred_scaled)
        
        y_test_true = self.preprocessor.inverse_transform_predictions(y_test)
        y_test_pred = self.preprocessor.inverse_transform_predictions(y_test_pred_scaled)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator()
        
        # Generate comprehensive report
        train_metrics, test_metrics = self.evaluator.generate_comprehensive_report(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
            self.history,
            output_dir=RESULTS_DIR

        )



        print("\n[STEP 3 COMPLETED] Evaluation finished successfully!")
        
        # ============================================================
        # STEP 4: SAVE RESULTS
        # ============================================================
        print("\n[STEP 4] SAVING RESULTS")
        print("-" * 70)
        
        # Save model
        self.tcn_model.save_model(os.path.join(MODELS_DIR, 'final_tcn_model.h5')
        )

        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual_Price': y_test_true,
            'Predicted_Price': y_test_pred,
            'Error': y_test_true - y_test_pred,
            'Absolute_Error': np.abs(y_test_true - y_test_pred),
            'Percentage_Error': np.abs((y_test_true - y_test_pred) / y_test_true) * 100
        })
        predictions_df.to_csv(os.path.join(RESULTS_DIR, 'final_predictions.csv'),index=False)

        print(f"Predictions saved to: {RESULTS_DIR}")

        
        print("\n[STEP 4 COMPLETED] All results saved successfully!")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "="*70)
        print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY ")
        print("="*70)
        print("\nFINAL RESULTS SUMMARY:")
        print("-" * 70)
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Test MAE: {test_metrics['MAE']:.4f}")
        print(f"Test R² Score: {test_metrics['R2']:.4f}")
        print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")
        print(f"Prediction Accuracy: {100 - test_metrics['MAPE']:.2f}%")
        print("-" * 70)
        
        print("\nOUTPUT FILES GENERATED:")
        print("  1. best_tcn_model.h5 - Best model during training")
        print("  2. final_tcn_model.h5 - Final trained model")
        print("  3. stock_analysis.png - Exploratory data analysis plots")
        print("  4. test_predictions.png - Actual vs Predicted plot")
        print("  5. training_history.png - Training metrics over epochs")
        print("  6. error_distribution.png - Error analysis plots")
        print("  7. residuals.png - Residual analysis")
        print("  8. final_predictions.csv - Detailed predictions")
        print("  9. predictions_comparison.csv - Sample comparisons")
        print("  10. evaluation_metrics.csv - All metrics")
        
        print("\n" + "="*70)
        print(" Thank you for using the Stock Price Forecasting System! ")
        print("="*70 + "\n")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': self.history,
            'predictions': predictions_df
        }


def main():
    """
    Main function to run the forecasting pipeline
    """
    # Configuration
    DATA_PATH = 'nifty50_data.csv'  # Change this to your data file path
    COMPANY_NAME = None  # None = use first company, or specify like 'ADANIPORTS'
    
    # Hyperparameters
    SEQUENCE_LENGTH = 60  # Number of past days to look at
    TRAIN_RATIO = 0.8     # 80% train, 20% test
    NUM_FILTERS = 64      # Number of filters in TCN
    KERNEL_SIZE = 3       # Kernel size for convolution
    NUM_BLOCKS = 4        # Number of temporal blocks
    EPOCHS = 100          # Number of training epochs
    BATCH_SIZE = 32       # Batch size
    LEARNING_RATE = 0.001 # Learning rate
    
    # Create forecaster
    forecaster = StockPriceForecaster(DATA_PATH, COMPANY_NAME)
    
    # Run pipeline
    results = forecaster.run_pipeline(
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        num_blocks=NUM_BLOCKS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    return results


if __name__ == "__main__":
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()
    
    # Run main function
    results = main()

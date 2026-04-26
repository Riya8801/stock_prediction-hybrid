"""
Model Evaluation and Visualization Module
Author: Your Name
Project: Forecasting Closing Prices using TCN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


class ModelEvaluator:
    """
    Class to evaluate and visualize model performance
    """
    
    def __init__(self):
        """
        Initialize the evaluator
        """
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred, set_name="Test"):
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            set_name: Name of the dataset (Train/Test)
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*50}")
        print(f"{set_name.upper()} SET EVALUATION METRICS")
        print(f"{'='*50}")
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        # Print metrics
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Calculate accuracy (custom metric for price prediction)
        # Accuracy = 100 - MAPE
        accuracy = 100 - mape
        print(f"Prediction Accuracy: {accuracy:.2f}%")
        
        self.metrics[set_name] = metrics
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, title="Stock Price Predictions", 
                        save_path='/home/claude/predictions_plot.png'):
        """
        Plot actual vs predicted values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 6))
        
        # Plot actual and predicted values
        plt.plot(y_true, label='Actual Price', linewidth=2, color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted Price', linewidth=2, color='red', alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPredictions plot saved to: {save_path}")
        plt.close()
    
    def plot_training_history(self, history, save_path='/home/claude/training_history.png'):
        """
        Plot training history
        
        Args:
            history: Training history object
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE
        axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Mean Absolute Error', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred, 
                               save_path='/home/claude/error_distribution.png'):
        """
        Plot error distribution
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold')
        
        # Histogram of errors
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_title('Error Distribution', fontweight='bold')
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Scatter plot: Actual vs Predicted
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                     label='Perfect Prediction')
        
        axes[1].set_title('Actual vs Predicted Prices', fontweight='bold')
        axes[1].set_xlabel('Actual Price')
        axes[1].set_ylabel('Predicted Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to: {save_path}")
        plt.close()
    
    def plot_residuals(self, y_true, y_pred, 
                      save_path='/home/claude/residuals_plot.png'):
        """
        Plot residuals over time
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(residuals, linewidth=1.5, color='purple', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.title('Residuals Over Time', fontweight='bold', fontsize=14)
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.scatter(range(len(residuals)), residuals, alpha=0.5, s=20, color='purple')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.title('Residual Scatter Plot', fontweight='bold', fontsize=14)
        plt.xlabel('Time Steps')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to: {save_path}")
        plt.close()
    
    def create_comparison_table(self, y_true, y_pred, num_samples=20):
        """
        Create a comparison table of actual vs predicted values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            num_samples: Number of samples to display
            
        Returns:
            DataFrame with comparison
        """
        # Select evenly spaced samples
        indices = np.linspace(0, len(y_true)-1, num_samples, dtype=int)
        
        comparison_df = pd.DataFrame({
            'Index': indices,
            'Actual Price': y_true[indices],
            'Predicted Price': y_pred[indices],
            'Error': y_true[indices] - y_pred[indices],
            'Absolute Error': np.abs(y_true[indices] - y_pred[indices]),
            'Percentage Error (%)': np.abs((y_true[indices] - y_pred[indices]) / y_true[indices]) * 100
        })
        
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def generate_comprehensive_report(self, y_train_true, y_train_pred, 
                                     y_test_true, y_test_pred, history,
                                     output_dir='/home/claude/'):
        """
        Generate a comprehensive evaluation report
        
        Args:
            y_train_true: True training values
            y_train_pred: Predicted training values
            y_test_true: True test values
            y_test_pred: Predicted test values
            history: Training history
            output_dir: Directory to save outputs
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train_true, y_train_pred, "Training")
        test_metrics = self.calculate_metrics(y_test_true, y_test_pred, "Test")
        
        # Generate plots
        self.plot_predictions(y_test_true, y_test_pred, 
                            "Test Set: Actual vs Predicted Prices",
                            f'{output_dir}test_predictions.png')
        
        self.plot_training_history(history, f'{output_dir}training_history.png')
        
        self.plot_error_distribution(y_test_true, y_test_pred,
                                    f'{output_dir}error_distribution.png')
        
        self.plot_residuals(y_test_true, y_test_pred,
                          f'{output_dir}residuals.png')
        
        # Create comparison table
        comparison_df = self.create_comparison_table(y_test_true, y_test_pred)
        comparison_df.to_csv(f'{output_dir}predictions_comparison.csv', index=False)
        print(f"\nComparison table saved to: {output_dir}predictions_comparison.csv")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE'],
            'Training': [train_metrics['MSE'], train_metrics['RMSE'], 
                        train_metrics['MAE'], train_metrics['R2'], train_metrics['MAPE']],
            'Test': [test_metrics['MSE'], test_metrics['RMSE'], 
                    test_metrics['MAE'], test_metrics['R2'], test_metrics['MAPE']]
        })
        metrics_df.to_csv(f'{output_dir}evaluation_metrics.csv', index=False)
        print(f"Metrics saved to: {output_dir}evaluation_metrics.csv")
        
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETED")
        print("="*60)
        
        return train_metrics, test_metrics


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("This module handles model evaluation and visualization")

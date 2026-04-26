
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from data_preprocessing import StockDataPreprocessor
from tcn_model import TCNModel
from hybrid_model import HybridTCNLSTM
from model_evaluation import ModelEvaluator

np.random.seed(42)
tf.random.set_seed(42)


class HybridModelComparison:
    """Train and compare TCN vs Hybrid models"""
    
    def __init__(self, data_path, company_name=None):
        self.data_path = data_path
        self.company_name = company_name
        self.preprocessor = None
        
    def run_complete_comparison(self, sequence_length=60, train_ratio=0.8,
                               num_filters=64, num_lstm_units=64,
                               epochs=50, batch_size=32, learning_rate=0.001):
        """Run complete pipeline with both models"""
        
        print("\n" + "="*80)
        print("  🚀 HYBRID TCN-LSTM vs TCN COMPARISON SYSTEM  ")
        print("="*80)
        print("\n📊 Dataset: NIFTY 50 Indian Stock Market")
        print("🎯 Goal: Compare Hybrid Model with Baseline TCN")
        
        # ==================== DATA PREPROCESSING ====================
        print("\n" + "="*80)
        print("[STEP 1] DATA PREPROCESSING")
        print("="*80)
        
        self.preprocessor = StockDataPreprocessor(self.data_path)
        data = self.preprocessor.load_data()
        self.preprocessor.explore_data()
        self.preprocessor.handle_missing_values()
        self.preprocessor.visualize_data(self.company_name)
        
        feature_data = self.preprocessor.prepare_features(self.company_name)
        scaled_data = self.preprocessor.scale_data()
        X, y = self.preprocessor.create_sequences(sequence_length=sequence_length)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, train_ratio)
        
        print("\n✅ Data preprocessing completed!")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # ==================== MODEL 1: TCN (BASELINE) ====================
        print("\n" + "="*80)
        print("[STEP 2] TRAINING BASELINE TCN MODEL")
        print("="*80)
        
        tcn_model = TCNModel(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=3,
            num_blocks=4,
            dropout_rate=0.2
        )
        
        print("\n🔷 Building TCN Model...")
        tcn_model.build_model()
        tcn_model.compile_model(learning_rate=learning_rate)
        
        print("\n🏃 Training TCN Model...")
        tcn_history = tcn_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size,
            model_save_path='best_tcn_baseline.h5'
        )
        
        # TCN predictions
        y_train_pred_tcn_scaled = tcn_model.predict(X_train)
        y_test_pred_tcn_scaled = tcn_model.predict(X_test)
        
        y_train_true = self.preprocessor.inverse_transform_predictions(y_train)
        y_train_pred_tcn = self.preprocessor.inverse_transform_predictions(y_train_pred_tcn_scaled)
        y_test_true = self.preprocessor.inverse_transform_predictions(y_test)
        y_test_pred_tcn = self.preprocessor.inverse_transform_predictions(y_test_pred_tcn_scaled)
        
        print("\n✅ TCN Model training completed!")
        
        # ==================== MODEL 2: HYBRID TCN-LSTM ====================
        print("\n" + "="*80)
        print("[STEP 3] TRAINING HYBRID TCN-LSTM MODEL")
        print("="*80)
        
        hybrid_model = HybridTCNLSTM(
            input_shape=input_shape,
            num_tcn_filters=num_filters,
            num_lstm_units=num_lstm_units,
            kernel_size=3,
            num_blocks=3,
            dropout_rate=0.2
        )
        
        print("\n🔶 Building Hybrid Model...")
        hybrid_model.build_model()
        hybrid_model.compile_model(learning_rate=learning_rate)
        hybrid_model.get_summary()
        
        print("\n🏃 Training Hybrid Model...")
        hybrid_history = hybrid_model.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size,
            model_save_path='best_hybrid_model.h5'
        )
        
        # Hybrid predictions
        y_train_pred_hybrid_scaled = hybrid_model.predict(X_train)
        y_test_pred_hybrid_scaled = hybrid_model.predict(X_test)
        
        y_train_pred_hybrid = self.preprocessor.inverse_transform_predictions(y_train_pred_hybrid_scaled)
        y_test_pred_hybrid = self.preprocessor.inverse_transform_predictions(y_test_pred_hybrid_scaled)
        
        print("\n✅ Hybrid Model training completed!")
        
        # ==================== EVALUATION & COMPARISON ====================
        print("\n" + "="*80)
        print("[STEP 4] MODEL EVALUATION & COMPARISON")
        print("="*80)
        
        evaluator = ModelEvaluator()
        
        # Evaluate both models
        print("\n📊 Evaluating Models...")
        tcn_metrics = evaluator.calculate_metrics(y_test_true, y_test_pred_tcn, "TCN Model")
        hybrid_metrics = evaluator.calculate_metrics(y_test_true, y_test_pred_hybrid, "Hybrid Model")
        
        # Create comparison visualizations
        print("\n📈 Creating comparison plots...")
        self._create_comparison_plots(
            y_test_true, y_test_pred_tcn, y_test_pred_hybrid,
            tcn_history, hybrid_history
        )
        
        # Save detailed results
        self._save_results(
            tcn_metrics, hybrid_metrics,
            y_test_true, y_test_pred_tcn, y_test_pred_hybrid
        )
        
        # Print final comparison
        self._print_comparison_summary(tcn_metrics, hybrid_metrics)
        
        print("\n✅ All steps completed successfully!")
        
        return {
            'tcn_metrics': tcn_metrics,
            'hybrid_metrics': hybrid_metrics,
            'tcn_history': tcn_history,
            'hybrid_history': hybrid_history
        }
    
    def _create_comparison_plots(self, y_true, y_pred_tcn, y_pred_hybrid, 
                                 tcn_hist, hybrid_hist):
        """Create comprehensive comparison plots"""
        
        # Figure 1: Main Comparison (4 subplots)
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Hybrid TCN-LSTM vs TCN: Comprehensive Comparison', 
                     fontsize=20, fontweight='bold', y=0.995)
        
        # Plot 1: Predictions Comparison
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(y_true, label='Actual Price', linewidth=2.5, color='black', alpha=0.8)
        ax1.plot(y_pred_tcn, label='TCN Prediction', linewidth=2, color='#3498db', 
                alpha=0.7, linestyle='--')
        ax1.plot(y_pred_hybrid, label='Hybrid Prediction', linewidth=2, color='#e74c3c', 
                alpha=0.7)
        ax1.set_title('Price Predictions Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps', fontsize=11)
        ax1.set_ylabel('Stock Price (₹)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Loss Comparison
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(tcn_hist.history['loss'], label='TCN Training Loss', 
                linewidth=2, color='#3498db', alpha=0.7)
        ax2.plot(hybrid_hist.history['loss'], label='Hybrid Training Loss', 
                linewidth=2, color='#e74c3c', alpha=0.7)
        ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss (MSE)', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation Loss Comparison
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(tcn_hist.history['val_loss'], label='TCN Val Loss', 
                linewidth=2, color='#3498db')
        ax3.plot(hybrid_hist.history['val_loss'], label='Hybrid Val Loss', 
                linewidth=2, color='#e74c3c')
        ax3.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Validation Loss (MSE)', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: TCN Error Distribution
        ax4 = plt.subplot(2, 3, 4)
        tcn_errors = y_true - y_pred_tcn
        ax4.hist(tcn_errors, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax4.set_title('TCN Error Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Prediction Error (₹)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Hybrid Error Distribution
        ax5 = plt.subplot(2, 3, 5)
        hybrid_errors = y_true - y_pred_hybrid
        ax5.hist(hybrid_errors, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax5.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Zero Error')
        ax5.set_title('Hybrid Error Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Prediction Error (₹)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Error Comparison Scatter
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(abs(tcn_errors), abs(hybrid_errors), alpha=0.5, s=30, color='purple')
        max_error = max(abs(tcn_errors).max(), abs(hybrid_errors).max())
        ax6.plot([0, max_error], [0, max_error], 'r--', linewidth=2, label='Equal Error Line')
        ax6.set_title('Absolute Error Comparison', fontsize=14, fontweight='bold')
        ax6.set_xlabel('TCN Absolute Error (₹)', fontsize=11)
        ax6.set_ylabel('Hybrid Absolute Error (₹)', fontsize=11)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hybrid_vs_tcn_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Saved: hybrid_vs_tcn_comparison.png")
        plt.close()
        
        # Figure 2: Metrics Bar Chart
        self._plot_metrics_bars()
    
    def _plot_metrics_bars(self):
        """Create bar chart comparing metrics"""
        # This will be populated with actual metrics in _save_results
        pass
    
    def _save_results(self, tcn_metrics, hybrid_metrics, y_true, y_pred_tcn, y_pred_hybrid):
        """Save detailed comparison results"""
        
        # Save comparison metrics
        comparison_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²', 'MAPE'],
            'TCN': [tcn_metrics['RMSE'], tcn_metrics['MAE'], 
                   tcn_metrics['R2'], tcn_metrics['MAPE']],
            'Hybrid': [hybrid_metrics['RMSE'], hybrid_metrics['MAE'], 
                      hybrid_metrics['R2'], hybrid_metrics['MAPE']],
            'Improvement': [
                ((tcn_metrics['RMSE'] - hybrid_metrics['RMSE']) / tcn_metrics['RMSE'] * 100),
                ((tcn_metrics['MAE'] - hybrid_metrics['MAE']) / tcn_metrics['MAE'] * 100),
                ((hybrid_metrics['R2'] - tcn_metrics['R2']) / tcn_metrics['R2'] * 100),
                ((tcn_metrics['MAPE'] - hybrid_metrics['MAPE']) / tcn_metrics['MAPE'] * 100)
            ]
        })
        comparison_df.to_csv('model_comparison_metrics.csv', index=False)
        print("💾 Saved: model_comparison_metrics.csv")
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'Actual_Price': y_true,
            'TCN_Prediction': y_pred_tcn,
            'Hybrid_Prediction': y_pred_hybrid,
            'TCN_Error': y_true - y_pred_tcn,
            'Hybrid_Error': y_true - y_pred_hybrid,
            'TCN_Abs_Error': np.abs(y_true - y_pred_tcn),
            'Hybrid_Abs_Error': np.abs(y_true - y_pred_hybrid)
        })
        predictions_df.to_csv('hybrid_detailed_predictions.csv', index=False)
        print("💾 Saved: hybrid_detailed_predictions.csv")
        
        # Create metrics bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparison_df['TCN'], width, label='TCN', 
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, comparison_df['Hybrid'], width, label='Hybrid', 
                      color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Metric'])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('metrics_comparison_chart.png', dpi=300, bbox_inches='tight')
        print("📊 Saved: metrics_comparison_chart.png")
        plt.close()
    
    def _print_comparison_summary(self, tcn_metrics, hybrid_metrics):
        """Print final comparison summary"""
        print("\n" + "="*80)
        print("  📊 FINAL COMPARISON SUMMARY  ")
        print("="*80)
        
        print("\n🔷 TCN MODEL RESULTS:")
        print(f"   RMSE: ₹{tcn_metrics['RMSE']:.4f}")
        print(f"   MAE:  ₹{tcn_metrics['MAE']:.4f}")
        print(f"   R²:   {tcn_metrics['R2']:.4f}")
        print(f"   MAPE: {tcn_metrics['MAPE']:.2f}%")
        print(f"   Accuracy: {100 - tcn_metrics['MAPE']:.2f}%")
        
        print("\n🔶 HYBRID MODEL RESULTS:")
        print(f"   RMSE: ₹{hybrid_metrics['RMSE']:.4f}")
        print(f"   MAE:  ₹{hybrid_metrics['MAE']:.4f}")
        print(f"   R²:   {hybrid_metrics['R2']:.4f}")
        print(f"   MAPE: {hybrid_metrics['MAPE']:.2f}%")
        print(f"   Accuracy: {100 - hybrid_metrics['MAPE']:.2f}%")
        
        print("\n📈 IMPROVEMENT (Hybrid vs TCN):")
        rmse_imp = ((tcn_metrics['RMSE'] - hybrid_metrics['RMSE']) / tcn_metrics['RMSE'] * 100)
        mae_imp = ((tcn_metrics['MAE'] - hybrid_metrics['MAE']) / tcn_metrics['MAE'] * 100)
        r2_imp = ((hybrid_metrics['R2'] - tcn_metrics['R2']) / tcn_metrics['R2'] * 100)
        mape_imp = ((tcn_metrics['MAPE'] - hybrid_metrics['MAPE']) / tcn_metrics['MAPE'] * 100)
        
        print(f"   RMSE: {rmse_imp:+.2f}% {'✅ Better' if rmse_imp > 0 else '❌ Worse'}")
        print(f"   MAE:  {mae_imp:+.2f}% {'✅ Better' if mae_imp > 0 else '❌ Worse'}")
        print(f"   R²:   {r2_imp:+.2f}% {'✅ Better' if r2_imp > 0 else '❌ Worse'}")
        print(f"   MAPE: {mape_imp:+.2f}% {'✅ Better' if mape_imp > 0 else '❌ Worse'}")
        
        print("\n" + "="*80)
        print("📁 OUTPUT FILES GENERATED:")
        print("   1. best_tcn_baseline.h5 - TCN model")
        print("   2. best_hybrid_model.h5 - Hybrid model")
        print("   3. hybrid_vs_tcn_comparison.png - Comparison plots")
        print("   4. metrics_comparison_chart.png - Metrics bar chart")
        print("   5. model_comparison_metrics.csv - Metrics comparison")
        print("   6. hybrid_detailed_predictions.csv - Detailed predictions")
        print("   7. stock_analysis.png - Data analysis")
        print("="*80)


def main():
    """Main function"""
    # ========== CONFIGURATION ==========
    DATA_PATH = 'nifty50_data.csv'  # Your CSV file
    COMPANY_NAME = None              # None or 'ADANIPORTS', 'RELIANCE', etc.
    
    # Hyperparameters
    SEQUENCE_LENGTH = 60
    TRAIN_RATIO = 0.8
    NUM_FILTERS = 128
    NUM_LSTM_UNITS = 128
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # ========== RUN COMPARISON ==========
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
    
    comparison = HybridModelComparison(DATA_PATH, COMPANY_NAME)
    results = comparison.run_complete_comparison(
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
        num_filters=NUM_FILTERS,
        num_lstm_units=NUM_LSTM_UNITS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ SUCCESS! Hybrid model comparison completed.")
        print("\n💡 Next: Check the generated PNG and CSV files!")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 Make sure your CSV file is in the same folder!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 Check error message and troubleshoot.")

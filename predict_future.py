"""
Future Price Prediction Script
Use trained model to predict future stock prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from data_preprocessing import StockDataPreprocessor
from tcn_model import TCNModel, TemporalBlock


class FuturePricePredictor:
    """
    Class to predict future stock prices using trained TCN model
    """
    
    def __init__(self, model_path, data_path, company_name=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model (.h5 file)
            data_path: Path to historical data CSV
            company_name: Company to predict (None = first company)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.company_name = company_name
        self.model = None
        self.preprocessor = None
        self.last_sequence = None
        
    def load_model_and_data(self, sequence_length=60):
        """
        Load the trained model and prepare data
        
        Args:
            sequence_length: Sequence length used during training
        """
        print("Loading model and data...")
        
        # Load model with custom objects
        custom_objects = {'TemporalBlock': TemporalBlock}
        self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
        print(f"Model loaded from: {self.model_path}")
        
        # Load and preprocess data
        self.preprocessor = StockDataPreprocessor(self.data_path)
        data = self.preprocessor.load_data()
        self.preprocessor.handle_missing_values()
        features = self.preprocessor.prepare_features(self.company_name)
        scaled_data = self.preprocessor.scale_data()
        
        # Get last sequence for prediction
        self.last_sequence = scaled_data[-sequence_length:]
        
        print(f"Data prepared. Last sequence shape: {self.last_sequence.shape}")
        
    def predict_next_day(self):
        """
        Predict the next day's closing price
        
        Returns:
            Predicted closing price
        """
        # Reshape for model input
        input_data = self.last_sequence.reshape(1, self.last_sequence.shape[0], 
                                                self.last_sequence.shape[1])
        
        # Make prediction (scaled)
        prediction_scaled = self.model.predict(input_data, verbose=0)[0][0]
        
        # Inverse transform to get actual price
        prediction = self.preprocessor.inverse_transform_predictions(
            np.array([prediction_scaled])
        )[0]
        
        return prediction
    
    def predict_multiple_days(self, num_days=7):
        """
        Predict multiple days into the future
        
        Args:
            num_days: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        print(f"\nPredicting next {num_days} days...")
        
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for day in range(num_days):
            # Reshape for model input
            input_data = current_sequence.reshape(1, current_sequence.shape[0], 
                                                  current_sequence.shape[1])
            
            # Predict (scaled)
            pred_scaled = self.model.predict(input_data, verbose=0)[0][0]
            
            # Inverse transform
            pred_price = self.preprocessor.inverse_transform_predictions(
                np.array([pred_scaled])
            )[0]
            
            predictions.append(pred_price)
            
            # Update sequence for next prediction
            # Create new row with predicted values
            # For simplicity, we'll use the predicted close for all fields
            # In reality, you might want to predict all features or use different logic
            new_row = current_sequence[-1].copy()
            new_row[3] = pred_scaled  # Update close price (index 3)
            
            # Slide the sequence
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            print(f"Day {day+1}: ₹{pred_price:.2f}")
        
        return np.array(predictions)
    
    def visualize_predictions(self, predictions, save_path='/home/claude/future_predictions.png'):
        """
        Visualize future predictions with historical data
        
        Args:
            predictions: Array of predicted prices
            save_path: Path to save the plot
        """
        # Get recent historical data
        data = self.preprocessor.data
        if self.company_name:
            company_data = data[data['Company'] == self.company_name]
        else:
            company_data = data[data['Company'] == data['Company'].iloc[0]]
        
        recent_data = company_data.tail(60)
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot historical data
        plt.plot(range(len(recent_data)), recent_data['Close'].values, 
                label='Historical Prices', linewidth=2, color='blue', marker='o', markersize=3)
        
        # Plot predictions
        future_x = range(len(recent_data), len(recent_data) + len(predictions))
        plt.plot(future_x, predictions, 
                label='Predicted Prices', linewidth=2, color='red', 
                linestyle='--', marker='s', markersize=6)
        
        # Add vertical line to separate history from predictions
        plt.axvline(x=len(recent_data)-1, color='green', linestyle=':', linewidth=2, 
                   label='Prediction Start')
        
        # Formatting
        plt.title('Stock Price Forecast', fontsize=16, fontweight='bold')
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Closing Price (₹)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Annotate predictions
        for i, pred in enumerate(predictions):
            plt.annotate(f'₹{pred:.2f}', 
                        xy=(len(recent_data) + i, pred),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrediction plot saved to: {save_path}")
        plt.close()
    
    def save_predictions(self, predictions, save_path='/home/claude/future_predictions.csv'):
        """
        Save predictions to CSV file
        
        Args:
            predictions: Array of predicted prices
            save_path: Path to save CSV
        """
        # Get last date from data
        data = self.preprocessor.data
        if self.company_name:
            company_data = data[data['Company'] == self.company_name]
        else:
            company_data = data[data['Company'] == data['Company'].iloc[0]]
        
        last_date = pd.to_datetime(company_data['Date'].iloc[-1])
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=len(predictions), freq='D')
        
        # Create DataFrame
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close_Price': predictions,
            'Day': [f'Day {i+1}' for i in range(len(predictions))]
        })
        
        # Save to CSV
        predictions_df.to_csv(save_path, index=False)
        print(f"Predictions saved to: {save_path}")
        
        return predictions_df


def main():
    """
    Main function to run future price prediction
    """
    print("="*70)
    print("        FUTURE STOCK PRICE PREDICTION USING TRAINED TCN MODEL")
    print("="*70)
    
    # Configuration
    MODEL_PATH = '/home/claude/best_tcn_model.h5'  # Path to trained model
    DATA_PATH = 'stock_data.csv'                    # Path to data
    COMPANY_NAME = None                             # Company name (None = first)
    SEQUENCE_LENGTH = 60                            # Must match training
    NUM_DAYS_TO_PREDICT = 7                        # Days to forecast
    
    # Create predictor
    predictor = FuturePricePredictor(MODEL_PATH, DATA_PATH, COMPANY_NAME)
    
    # Load model and data
    predictor.load_model_and_data(sequence_length=SEQUENCE_LENGTH)
    
    # Predict next day
    next_day_price = predictor.predict_next_day()
    print(f"\n{'='*70}")
    print(f"NEXT DAY PREDICTION")
    print(f"{'='*70}")
    print(f"Predicted closing price for tomorrow: ₹{next_day_price:.2f}")
    
    # Predict multiple days
    print(f"\n{'='*70}")
    print(f"MULTI-DAY PREDICTIONS")
    print(f"{'='*70}")
    predictions = predictor.predict_multiple_days(num_days=NUM_DAYS_TO_PREDICT)
    
    # Visualize
    predictor.visualize_predictions(predictions)
    
    # Save predictions
    predictions_df = predictor.save_predictions(predictions)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"Average predicted price: ₹{predictions.mean():.2f}")
    print(f"Highest predicted price: ₹{predictions.max():.2f}")
    print(f"Lowest predicted price: ₹{predictions.min():.2f}")
    print(f"Price range: ₹{predictions.max() - predictions.min():.2f}")
    
    # Trend analysis
    trend = "upward" if predictions[-1] > predictions[0] else "downward"
    change_percent = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
    print(f"\nOverall trend: {trend.upper()}")
    print(f"Expected change: {change_percent:+.2f}%")
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")
    
    return predictions_df


if __name__ == "__main__":
    predictions_df = main()
    print("\nPredicted prices for next 7 days:")
    print(predictions_df.to_string(index=False))

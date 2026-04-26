"""
Data Preprocessing Module for Stock Price Forecasting
Author: Your Name
Project: Forecasting Closing Prices using TCN
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

class StockDataPreprocessor:
    """
    Class to handle all data preprocessing tasks
    """
    
    def __init__(self, filepath):
        """
        Initialize the preprocessor with the dataset filepath
        
        Args:
            filepath: Path to the CSV file containing stock data
        """
        self.filepath = filepath
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def load_data(self):
        """
        Load the stock market data from CSV
        """
        print("Loading data...")
        self.data = pd.read_csv(self.filepath)
        
        # Convert Date column to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='mixed')

        
        # Sort by date
        self.data = self.data.sort_values('Date')
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        print(f"\nData info:")
        print(self.data.info())
        
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Check for duplicates
        print(f"\nDuplicate rows: {self.data.duplicated().sum()}")
        
        # Company information
        print(f"\nUnique companies: {self.data['Company'].nunique()}")
        print(f"Companies: {self.data['Company'].unique()}")
        
        return self.data.describe()
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("\nHandling missing values...")
        
        # Forward fill for missing values
        self.data = self.data.fillna(method='ffill')
        
        # Backward fill for any remaining NaN
        self.data = self.data.fillna(method='bfill')
        
        print(f"Missing values after handling: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def visualize_data(self, company_name=None):
        """
        Create visualizations for the stock data
        
        Args:
            company_name: Name of company to visualize (if None, uses first company)
        """
        if company_name is None:
            company_name = self.data['Company'].iloc[0]
        
        # Filter data for specific company
        company_data = self.data[self.data['Company'] == company_name].copy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Stock Analysis for {company_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Closing Price Over Time
        axes[0, 0].plot(company_data['Date'], company_data['Close'], linewidth=2, color='blue')
        axes[0, 0].set_title('Closing Price Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Closing Price')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Volume Over Time
        axes[0, 1].bar(company_data['Date'], company_data['Volume'], color='green', alpha=0.6)
        axes[0, 1].set_title('Trading Volume Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: High, Low, Close
        axes[1, 0].plot(company_data['Date'], company_data['High'], label='High', linewidth=1.5, alpha=0.7)
        axes[1, 0].plot(company_data['Date'], company_data['Low'], label='Low', linewidth=1.5, alpha=0.7)
        axes[1, 0].plot(company_data['Date'], company_data['Close'], label='Close', linewidth=2)
        axes[1, 0].set_title('High, Low, Close Prices', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Distribution of Closing Prices
        axes[1, 1].hist(company_data['Close'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Closing Prices', fontweight='bold')
        axes[1, 1].set_xlabel('Closing Price')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('stock_analysis.png', dpi=300, bbox_inches='tight')

        print(f"\nVisualization saved as 'stock_analysis.png'")
        plt.close()
        
    def prepare_features(self, company_name=None):
        """
        Prepare features for the model
        
        Args:
            company_name: Name of company to process (if None, uses first company)
        """
        if company_name is None:
            company_name = self.data['Company'].iloc[0]
        
        print(f"\nPreparing features for {company_name}...")
        
        # Filter data for specific company
        company_data = self.data[self.data['Company'] == company_name].copy()
        
        # Add Technical Indicators
        print("Calculating Technical Indicators (SMA, RSI, MACD)...")
        # SMA
        company_data['SMA_10'] = company_data['Close'].rolling(window=10).mean()
        company_data['SMA_50'] = company_data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = company_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        company_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = company_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = company_data['Close'].ewm(span=26, adjust=False).mean()
        company_data['MACD'] = exp1 - exp2
        
        # Fill NaN values created by rolling windows
        company_data = company_data.fillna(method='bfill')

        # Select features
        # We'll use: Open, High, Low, Close, Volume + Technical Indicators
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD']
        self.feature_data = company_data[features].values
        
        print(f"Feature data shape: {self.feature_data.shape}")
        print(f"Features used: {features}")
        
        return self.feature_data
    
    def scale_data(self):
        """
        Scale the data using MinMaxScaler
        """
        print("\nScaling data...")
        self.scaled_data = self.scaler.fit_transform(self.feature_data)
        print(f"Scaled data shape: {self.scaled_data.shape}")
        print(f"Scaled data range: [{self.scaled_data.min()}, {self.scaled_data.max()}]")
        
        return self.scaled_data
    
    def create_sequences(self, sequence_length=60):
        """
        Create sequences for time series prediction
        
        Args:
            sequence_length: Number of time steps to look back
            
        Returns:
            X: Input sequences
            y: Target values (closing prices)
        """
        print(f"\nCreating sequences with sequence_length={sequence_length}...")
        
        X, y = [], []
        
        for i in range(sequence_length, len(self.scaled_data)):
            # Input: all features for past 'sequence_length' days
            X.append(self.scaled_data[i-sequence_length:i, :])
            
            # Output: closing price of the next day (index 3 is Close)
            y.append(self.scaled_data[i, 3])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.8):
        """
        Split data into training and testing sets
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of training data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nSplitting data with train_ratio={train_ratio}...")
        
        split_index = int(len(X) * train_ratio)
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Testing set - X: {X_test.shape}, y: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform the predictions to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Original scale predictions
        """
        # Create a dummy array with same shape as original features
        dummy = np.zeros((len(predictions), self.feature_data.shape[1]))
        
        # Place predictions in the 'Close' column (index 3)
        dummy[:, 3] = predictions
        
        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)
        
        # Return only the Close prices
        return inversed[:, 3]


if __name__ == "__main__":
    # Example usage
    print("Stock Data Preprocessor Module")
    print("This module handles data loading, preprocessing, and preparation for TCN model")

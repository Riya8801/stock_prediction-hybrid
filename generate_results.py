#!/usr/bin/env python3
"""
Generate comprehensive visualization and results for Stock Prediction Hybrid System
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# Paths
# ============================================================================

OUTPUT_DIR = "output"
DATA_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("📊 GENERATING COMPREHENSIVE RESULTS AND VISUALIZATIONS")
print("=" * 80)
print()

# ============================================================================
# Step 1: Load Data
# ============================================================================

print("Step 1: Loading data...")

df_data = None
df_predictions = None
df_metrics = None

try:
    df_data = pd.read_csv(os.path.join(DATA_DIR, 'nifty50_data.csv'))
    print(f"✅ Loaded NIFTY50 data: {df_data.shape[0]} rows")
except Exception as e:
    print(f"⚠️  Could not load data: {e}")

try:
    df_predictions = pd.read_csv(os.path.join(OUTPUT_DIR, 'hybrid_detailed_predictions.csv'))
    print(f"✅ Loaded predictions: {df_predictions.shape[0]} records")
except Exception as e:
    print(f"⚠️  Could not load predictions: {e}")

try:
    df_metrics = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_comparison_metrics.csv'))
    print(f"✅ Loaded metrics: {df_metrics.shape}")
except Exception as e:
    print(f"⚠️  Could not load metrics: {e}")

print()

# ============================================================================
# Step 2: Generate Stock Price Analysis
# ============================================================================

if df_data is not None and len(df_data) > 0:
    print("Step 2: Generating stock price analysis...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Stock Price Analysis - NIFTY 50', fontsize=16, fontweight='bold')
        
        # Use Close price
        prices = pd.to_numeric(df_data['Close'], errors='coerce').dropna()
        
        # Plot 1: Price trend
        axes[0, 0].plot(range(len(prices)), prices, linewidth=2, color='steelblue')
        axes[0, 0].set_title('Stock Price Trend', fontweight='bold')
        axes[0, 0].set_xlabel('Time Period')
        axes[0, 0].set_ylabel('Price (₹)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Daily returns
        returns = prices.pct_change().dropna() * 100
        axes[0, 1].plot(range(len(returns)), returns, linewidth=1, color='coral', alpha=0.7)
        axes[0, 1].set_title('Daily Returns Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('Returns (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility
        rolling_vol = returns.rolling(window=20).std()
        axes[1, 0].fill_between(range(len(rolling_vol)), rolling_vol, alpha=0.5, color='orange')
        axes[1, 0].plot(range(len(rolling_vol)), rolling_vol, color='darkorange', linewidth=2)
        axes[1, 0].set_title('Rolling 20-Day Volatility', fontweight='bold')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Distribution
        axes[1, 1].hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Returns Distribution (Histogram)', fontweight='bold')
        axes[1, 1].set_xlabel('Daily Returns (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, 'stock_price_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Generated: stock_price_analysis.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Error generating stock price analysis: {e}")
    
    print()

# ============================================================================
# Step 3: Model Performance Metrics
# ============================================================================

if df_metrics is not None and len(df_metrics) > 0:
    print("Step 3: Generating model comparison metrics...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Get model names and metrics
        model_names = df_metrics.iloc[:, 0]
        metrics_cols = df_metrics.columns[1:]
        
        col_idx = 0
        for i, col in enumerate(metrics_cols[:4]):
            row = i // 2
            col_num = i % 2
            
            try:
                values = pd.to_numeric(df_metrics[col], errors='coerce').dropna()
                if len(values) > 0:
                    axes[row, col_num].bar(range(len(values)), values, 
                                          color=['gold', 'silver', 'orange', 'coral'][:len(values)])
                    axes[row, col_num].set_title(f'{col} Comparison', fontweight='bold')
                    axes[row, col_num].set_ylabel(col)
                    for j, v in enumerate(values):
                        axes[row, col_num].text(j, v + max(values)*0.02, f'{v:.2f}', ha='center')
                    axes[row, col_num].grid(True, alpha=0.3, axis='y')
            except:
                pass
        
        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, 'model_performance_metrics.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Generated: model_performance_metrics.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Error generating model metrics: {e}")
    
    print()

# ============================================================================
# Step 4: Predictions Visualization
# ============================================================================

if df_predictions is not None and len(df_predictions) > 0:
    print("Step 4: Generating predictions visualization...")
    
    try:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Model Predictions Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Full series
        x_range = range(len(df_predictions))
        axes[0].plot(x_range, df_predictions.iloc[:, 0], linewidth=2, marker='o', 
                    markersize=3, alpha=0.7, label='Column 1', color='steelblue')
        if len(df_predictions.columns) > 1:
            axes[0].plot(x_range, df_predictions.iloc[:, 1], linewidth=2, linestyle='--',
                        alpha=0.7, label='Column 2', color='coral')
        axes[0].set_title('Full Prediction Series', fontweight='bold')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Last 50 points
        start_idx = max(0, len(df_predictions) - 50)
        x_subset = range(start_idx, len(df_predictions))
        axes[1].plot(x_subset, df_predictions.iloc[start_idx:, 0], linewidth=2, marker='o',
                    markersize=6, alpha=0.7, label='Column 1', color='steelblue')
        if len(df_predictions.columns) > 1:
            axes[1].plot(x_subset, df_predictions.iloc[start_idx:, 1], linewidth=2, linestyle='--',
                        alpha=0.7, label='Column 2', color='coral')
        axes[1].set_title('Last 50 Predictions (Detailed)', fontweight='bold')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, 'predictions_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Generated: predictions_visualization.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Error generating predictions: {e}")
    
    print()

# ============================================================================
# Step 5: Create Summary Report
# ============================================================================

print("Step 5: Creating summary report...")

try:
    summary_file = os.path.join(OUTPUT_DIR, 'RESULTS_SUMMARY.txt')
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STOCK PREDICTION HYBRID SYSTEM - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if df_data is not None and len(df_data) > 0:
            f.write("DATA OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Records: {len(df_data)}\n")
            f.write(f"Columns: {list(df_data.columns)}\n\n")
        
        if df_metrics is not None and len(df_metrics) > 0:
            f.write("MODEL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(df_metrics.to_string())
            f.write("\n\n")
        
        if df_predictions is not None and len(df_predictions) > 0:
            f.write("PREDICTION SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Predictions: {len(df_predictions)}\n")
            f.write(f"Columns: {list(df_predictions.columns)}\n\n")
        
        f.write("GENERATED FILES\n")
        f.write("-" * 80 + "\n")
        f.write("✅ stock_price_analysis.png\n")
        f.write("✅ model_performance_metrics.png\n")
        f.write("✅ predictions_visualization.png\n")
        f.write("✅ RESULTS_SUMMARY.txt\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✅ Generated: RESULTS_SUMMARY.txt")
except Exception as e:
    print(f"⚠️  Error creating summary: {e}")

print()

print("=" * 80)
print("✅ RESULTS GENERATION COMPLETE!")
print("=" * 80)

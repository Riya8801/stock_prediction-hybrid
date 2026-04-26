"""
Configuration File for Stock Price Forecasting
Edit these parameters to customize your model
"""

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

# Path to your CSV data file
DATA_PATH = 'stock_data.csv'

# Company name to analyze (set to None to use first company in dataset)
# Example: 'ADANIPORTS', 'RELIANCE', etc.
COMPANY_NAME = None

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

# Number of past days to look at for prediction
# Higher values = more historical context but slower training
# Recommended: 30-120
SEQUENCE_LENGTH = 60

# Number of filters in TCN convolutional layers
# Higher values = more model capacity but slower training
# Recommended: 32, 64, 128
NUM_FILTERS = 64

# Kernel size for convolution
# Affects the receptive field
# Recommended: 2-7
KERNEL_SIZE = 3

# Number of temporal blocks in TCN
# More blocks = deeper network, longer memory
# Recommended: 2-8
NUM_BLOCKS = 4

# Dropout rate for regularization
# Helps prevent overfitting
# Recommended: 0.1-0.3
DROPOUT_RATE = 0.2

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

# Ratio of data to use for training (rest is for testing)
# Recommended: 0.7-0.85
TRAIN_RATIO = 0.8

# Number of training epochs
# More epochs = more training time
# Recommended: 50-200
EPOCHS = 100

# Batch size for training
# Higher = faster training but needs more memory
# Recommended: 16, 32, 64
BATCH_SIZE = 32

# Learning rate for optimizer
# Too high = unstable training, too low = slow convergence
# Recommended: 0.0001-0.01
LEARNING_RATE = 0.001

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Directory to save outputs
OUTPUT_DIR = '/home/claude/'

# Model save path
MODEL_SAVE_PATH = OUTPUT_DIR + 'best_tcn_model.h5'
FINAL_MODEL_PATH = OUTPUT_DIR + 'final_tcn_model.h5'

# ==============================================================================
# ADVANCED SETTINGS (Usually don't need to change these)
# ==============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Enable GPU if available
USE_GPU = True

# Verbose level (0=silent, 1=progress bar, 2=one line per epoch)
VERBOSE = 1

# Early stopping patience (number of epochs to wait for improvement)
EARLY_STOPPING_PATIENCE = 15

# Reduce learning rate patience
REDUCE_LR_PATIENCE = 5

# Minimum learning rate
MIN_LEARNING_RATE = 1e-7

# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

# Uncomment one of these to use preset configurations

# # FAST TRAINING (for testing)
# SEQUENCE_LENGTH = 30
# NUM_BLOCKS = 2
# EPOCHS = 30
# BATCH_SIZE = 64

# # HIGH ACCURACY (takes longer)
# SEQUENCE_LENGTH = 90
# NUM_FILTERS = 128
# NUM_BLOCKS = 6
# EPOCHS = 150
# BATCH_SIZE = 16

# # BALANCED (default)
# SEQUENCE_LENGTH = 60
# NUM_FILTERS = 64
# NUM_BLOCKS = 4
# EPOCHS = 100
# BATCH_SIZE = 32

# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

# Features to use for prediction
# Available: 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# Target feature to predict
TARGET_FEATURE = 'Close'

# ==============================================================================
# VISUALIZATION SETTINGS
# ==============================================================================

# DPI for saved figures
FIGURE_DPI = 300

# Figure size
FIGURE_SIZE = (15, 6)

# Enable/disable plots
SAVE_PLOTS = True

# ==============================================================================
# NOTES FOR DIFFERENT SCENARIOS
# ==============================================================================

"""
FOR SHORT-TERM PREDICTION (next few days):
- SEQUENCE_LENGTH = 30-60
- NUM_BLOCKS = 3-4

FOR LONG-TERM PREDICTION (weeks/months):
- SEQUENCE_LENGTH = 90-120
- NUM_BLOCKS = 5-8

FOR LIMITED MEMORY:
- BATCH_SIZE = 16
- NUM_FILTERS = 32
- SEQUENCE_LENGTH = 30

FOR QUICK TESTING:
- EPOCHS = 20-30
- SEQUENCE_LENGTH = 30
- BATCH_SIZE = 64

FOR BEST ACCURACY (if you have time):
- SEQUENCE_LENGTH = 90
- NUM_FILTERS = 128
- NUM_BLOCKS = 6
- EPOCHS = 150
- BATCH_SIZE = 16
"""

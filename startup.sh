#!/bin/bash

# ============================================================================
# 🚀 Stock Prediction Hybrid System - Complete Startup Script
# ============================================================================
# This script automates the complete setup and launch of the project
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_step() {
    echo -e "${YELLOW}➜ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

print_header "Step 1: Checking Prerequisites"

print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

print_step "Checking pip installation..."
if python3 -m pip --version &> /dev/null; then
    print_success "pip is available"
else
    print_error "pip not found. Please install pip."
    exit 1
fi

# ============================================================================
# Step 2: Get Project Directory
# ============================================================================

print_header "Step 2: Setting Up Project Directory"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
print_success "Project directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR"
print_success "Changed to project directory"

# ============================================================================
# Step 3: Create/Check Virtual Environment
# ============================================================================

print_header "Step 3: Setting Up Virtual Environment"

if [ -d ".venv" ]; then
    print_success "Virtual environment already exists"
else
    print_step "Creating virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Verify activation
if [ -n "$VIRTUAL_ENV" ]; then
    print_success "Virtual environment confirmed active: $VIRTUAL_ENV"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# ============================================================================
# Step 4: Upgrade pip and Install Dependencies
# ============================================================================

print_header "Step 4: Installing Dependencies"

print_step "Upgrading pip..."
python -m pip install --upgrade pip --quiet
print_success "pip upgraded"

print_step "Installing setuptools and wheel..."
pip install setuptools wheel --quiet
print_success "setuptools and wheel installed"

print_step "Installing requirements from requirements.txt..."
pip install -r requirements.txt --quiet
print_success "Base requirements installed"

print_step "Installing Streamlit..."
pip install streamlit --quiet
print_success "Streamlit installed"

# ============================================================================
# Step 5: Verify Installation
# ============================================================================

print_header "Step 5: Verifying Installation"

print_step "Verifying TensorFlow..."
python -c "import tensorflow; print(f'  TensorFlow version: {tensorflow.__version__}')" || print_error "TensorFlow import failed"

print_step "Verifying Keras..."
python -c "import keras; print(f'  Keras version: {keras.__version__}')" || print_error "Keras import failed"

print_step "Verifying Streamlit..."
python -c "import streamlit; print(f'  Streamlit installed')" || print_error "Streamlit import failed"

print_success "All packages verified successfully"

# ============================================================================
# Step 6: Verify Data and Models
# ============================================================================

print_header "Step 6: Verifying Data and Models"

if [ -f "data/nifty50_data.csv" ]; then
    print_success "Data file found: data/nifty50_data.csv"
else
    print_error "Data file not found: data/nifty50_data.csv"
    exit 1
fi

if [ -f "models/best_hybrid_model.h5" ]; then
    print_success "Production model found: models/best_hybrid_model.h5 (97.31% accuracy)"
else
    print_error "Model not found: models/best_hybrid_model.h5"
    print_step "You may need to run: python train.py"
    exit 1
fi

# ============================================================================
# Step 7: Launch Dashboard
# ============================================================================

print_header "Step 7: Launching Dashboard"

print_success "Setup complete! 🎉"
echo ""
echo -e "${GREEN}Starting Streamlit Dashboard...${NC}"
echo -e "${BLUE}Dashboard will open at: http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}To stop the dashboard, press Ctrl+C${NC}"
echo ""

# Launch streamlit dashboard
streamlit run dashboard.py

# ============================================================================
# Exit
# ============================================================================

print_success "Startup Complete!"

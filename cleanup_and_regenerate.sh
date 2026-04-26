#!/bin/bash

# ============================================================================
# 🧹 Cleanup & Regenerate Results - Stock Prediction Hybrid System
# ============================================================================
# This script removes old result files and regenerates fresh visualizations
# ============================================================================

set -e

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
# Step 1: Navigate to Project Directory
# ============================================================================

print_header "Step 1: Setting Up"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
print_success "Project directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR"
print_success "Changed to project directory"

echo ""

# ============================================================================
# Step 2: Backup Important CSV Files
# ============================================================================

print_header "Step 2: Backing Up Data"

print_step "Creating backup of important CSV files..."

mkdir -p output/backups
BACKUP_DATE=$(date '+%Y%m%d_%H%M%S')

if [ -f "output/hybrid_detailed_predictions.csv" ]; then
    cp output/hybrid_detailed_predictions.csv "output/backups/hybrid_detailed_predictions_$BACKUP_DATE.csv"
    print_success "Backed up: hybrid_detailed_predictions.csv"
fi

if [ -f "output/model_comparison_metrics.csv" ]; then
    cp output/model_comparison_metrics.csv "output/backups/model_comparison_metrics_$BACKUP_DATE.csv"
    print_success "Backed up: model_comparison_metrics.csv"
fi

echo ""

# ============================================================================
# Step 3: Delete Old Image Files
# ============================================================================

print_header "Step 3: Removing Old Image Files"

print_step "Removing old PNG files..."

# Remove root directory images
rm -f *.png 2>/dev/null && print_success "Cleaned: root directory" || echo ""

# Remove output directory old images
rm -f output/*.png 2>/dev/null
rm -f output/results*.png 2>/dev/null
rm -f output/resultserror_distribution.png 2>/dev/null
rm -f output/resultstraining_history.png 2>/dev/null
rm -f output/resultsresiduals.png 2>/dev/null
rm -f output/resultstest_predictions.png 2>/dev/null

print_step "Removing old results files with 'results' prefix..."
rm -f output/results*.csv 2>/dev/null

print_success "Old image files cleaned"

echo ""

# ============================================================================
# Step 4: Activate Virtual Environment
# ============================================================================

print_header "Step 4: Activating Virtual Environment"

if [ -d ".venv" ]; then
    print_step "Found virtual environment, activating..."
    source .venv/bin/activate
    print_success "Virtual environment activated"
else
    print_error "Virtual environment not found"
    print_step "Creating new virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    print_step "Installing dependencies..."
    pip install -r requirements.txt --quiet
    pip install streamlit --quiet
    print_success "Setup complete"
fi

print_step "Installing required packages for result generation..."
pip install pandas numpy matplotlib seaborn scikit-learn scipy --quiet
print_success "Packages installed"

echo ""

# ============================================================================
# Step 5: Generate New Results
# ============================================================================

print_header "Step 5: Generating New Visualizations"

print_step "Running generate_results.py..."
echo ""

python generate_results.py

echo ""
print_success "Results generation complete"

echo ""

# ============================================================================
# Step 6: List Generated Files
# ============================================================================

print_header "Step 6: Generated Files Summary"

echo -e "${GREEN}Generated PNG Files:${NC}"
find output -name "*.png" -type f -exec ls -lh {} \; | awk '{print "  ✅ " $9 " (" $5 ")"}'

echo ""
echo -e "${GREEN}Data Files:${NC}"
find output -name "*.csv" -type f | grep -v backups | head -5 | awk '{print "  ✅ " $0}'

echo ""

# ============================================================================
# Completion
# ============================================================================

print_header "✅ Cleanup & Regeneration Complete"

echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  • Old image files removed"
echo "  • Data backed up to: output/backups/"
echo "  • New visualizations generated"
echo "  • Results saved to: output/"
echo ""

print_success "All results are now fresh and ready! 🚀"

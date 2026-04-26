#!/bin/bash

# ============================================================================
# 📤 Push to GitHub - Stock Prediction Hybrid System
# ============================================================================
# This script automates pushing the project to GitHub
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
# Step 1: Check Git Installation
# ============================================================================

print_header "Step 1: Checking Git Installation"

if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    print_success "Git found: $GIT_VERSION"
else
    print_error "Git not found. Please install Git."
    exit 1
fi

# ============================================================================
# Step 2: Get Project Directory
# ============================================================================

print_header "Step 2: Setting Up Project Directory"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
print_success "Project directory: $SCRIPT_DIR"

cd "$SCRIPT_DIR"
print_success "Changed to project directory"

# ============================================================================
# Step 3: Check if Git Repository Exists
# ============================================================================

print_header "Step 3: Checking Git Repository"

if [ -d ".git" ]; then
    print_success "Git repository found"
else
    print_step "Initializing git repository..."
    git init
    print_success "Git repository initialized"
fi

# ============================================================================
# Step 4: Configure Git (if needed)
# ============================================================================

print_header "Step 4: Configuring Git"

# Check if git user.name is set
if ! git config user.name &> /dev/null; then
    print_step "Git user.name not set. Setting to 'Developer'..."
    git config user.name "Developer"
fi

# Check if git user.email is set
if ! git config user.email &> /dev/null; then
    print_step "Git user.email not set. Setting to 'dev@example.com'..."
    git config user.email "dev@example.com"
fi

print_success "Git configured"

# ============================================================================
# Step 5: Check Remote and Add/Update
# ============================================================================

print_header "Step 5: Configuring Remote Repository"

GITHUB_URL="git@github.com:Riya8801/stock_prediction-hybrid.git"

if git remote | grep -q "^origin$"; then
    print_step "Remote 'origin' already exists. Updating..."
    git remote set-url origin "$GITHUB_URL"
    print_success "Remote updated: $GITHUB_URL"
else
    print_step "Adding remote repository..."
    git remote add origin "$GITHUB_URL"
    print_success "Remote added: $GITHUB_URL"
fi

# ============================================================================
# Step 6: Create .gitignore (if needed)
# ============================================================================

print_header "Step 6: Setting Up .gitignore"

if [ ! -f ".gitignore" ]; then
    print_step "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Virtual Environment
.venv/
venv/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
.temp/

# Node modules (if used)
node_modules/

# Environment variables
.env
.env.local
EOF
    print_success ".gitignore created"
else
    print_success ".gitignore already exists"
fi

# ============================================================================
# Step 7: Stage Changes
# ============================================================================

print_header "Step 7: Staging Changes"

print_step "Adding all files to git..."
git add -A
print_success "Files staged"

# ============================================================================
# Step 8: Check Status
# ============================================================================

print_header "Step 8: Checking Git Status"

echo ""
git status
echo ""

# ============================================================================
# Step 9: Commit Changes
# ============================================================================

print_header "Step 9: Creating Commit"

# Get current date for commit message
COMMIT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Check if there are changes to commit
if git diff --cached --quiet; then
    print_success "No changes to commit"
else
    print_step "Committing changes..."
    git commit -m "Update: Stock Prediction Hybrid System - $COMMIT_DATE

- Updated requirements.txt for Python 3.12 compatibility
- Enhanced startup.sh with improved error handling
- Added STARTUP.md documentation
- Updated dependencies and configurations"
    print_success "Changes committed"
fi

# ============================================================================
# Step 10: Get Current Branch
# ============================================================================

print_header "Step 10: Checking Current Branch"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
print_success "Current branch: $CURRENT_BRANCH"

# ============================================================================
# Step 11: Push to GitHub
# ============================================================================

print_header "Step 11: Pushing to GitHub"

print_step "Pushing to $GITHUB_URL (branch: $CURRENT_BRANCH)..."
echo ""

git push -u origin "$CURRENT_BRANCH"

print_success "Successfully pushed to GitHub! 🎉"

# ============================================================================
# Step 12: Display Results
# ============================================================================

print_header "Push Complete"

echo ""
echo -e "${GREEN}Repository Information:${NC}"
echo -e "  URL: ${YELLOW}$GITHUB_URL${NC}"
echo -e "  Branch: ${YELLOW}$CURRENT_BRANCH${NC}"
echo -e "  Commit: ${YELLOW}$(git rev-parse --short HEAD)${NC}"
echo ""
echo -e "${GREEN}View on GitHub:${NC}"
echo -e "  ${YELLOW}https://github.com/Riya8801/stock_prediction-hybrid${NC}"
echo ""

print_success "All done! Your code is now on GitHub 🚀"

#!/bin/bash
# train_script.sh - Sets up environment and trains the pronunciation classifier.

echo "--- Starting Classifier Training Setup ---"
set -e # Exit immediately if a command exits with a non-zero status.

# Ensure we are in the project root (or a known relative path)
# This script assumes it's run from the project root.
PROJECT_ROOT_DIR=$(pwd)
echo "Working from project root: $PROJECT_ROOT_DIR"

# Define venv directory for training scripts
VENV_SCRIPTS_DIR="$PROJECT_ROOT_DIR/venv_scripts"

if [ ! -d "$VENV_SCRIPTS_DIR" ]; then
    echo "Creating virtual environment for training scripts at $VENV_SCRIPTS_DIR..."
    python3 -m venv "$VENV_SCRIPTS_DIR"
else
    echo "Virtual environment for training scripts already exists at $VENV_SCRIPTS_DIR."
fi

echo "Activating training scripts virtual environment..."
source "$VENV_SCRIPTS_DIR/bin/activate"

echo "Installing dependencies for the training script (this may take a moment)..."
# For macOS ARM, ensure PyTorch picks up MPS compatible versions
pip install torch torchvision torchaudio
pip install pandas scikit-learn joblib Levenshtein

echo "Running classifier training script (scripts/train_classifier.py)..."
# Ensure the training script correctly locates backend/app/assets/ for input/output
python "$PROJECT_ROOT_DIR/scripts/train_classifier.py"

echo "Deactivating training scripts virtual environment..."
deactivate

echo "--- Classifier Training Script Finished ---"
echo "The model (phoneme_classifier.joblib) should now be in backend/app/assets/"
echo "You can find the virtual environment at $VENV_SCRIPTS_DIR"
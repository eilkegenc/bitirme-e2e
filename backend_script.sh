#!/bin/bash
# backend_script.sh - Sets up environment and runs the backend FastAPI server.

echo "--- Starting Backend Setup and Server ---"
set -e

# Navigate to the backend directory from the project root
PROJECT_ROOT_DIR=$(pwd)
BACKEND_DIR="$PROJECT_ROOT_DIR/backend"
CRISPER_WHISPER_CLONED_DIR="$PROJECT_ROOT_DIR/CrisperWhisper_cloned" # Path to cloned repo

if [ ! -d "$BACKEND_DIR" ]; then
    echo "Error: Backend directory not found at $BACKEND_DIR"
    echo "Please run this script from the root of the 'pronunciation_app' project."
    exit 1
fi

cd "$BACKEND_DIR"
echo "Changed directory to $(pwd)"

VENV_BACKEND_DIR="./venv_backend" # Relative to backend directory

if [ ! -d "$VENV_BACKEND_DIR" ]; then
    echo "Creating virtual environment for backend at $VENV_BACKEND_DIR..."
    python3 -m venv "$VENV_BACKEND_DIR"
else
    echo "Virtual environment for backend already exists at $VENV_BACKEND_DIR."
fi

echo "Activating backend virtual environment..."
source "$VENV_BACKEND_DIR/bin/activate"

echo "Installing CrisperWhisper dependencies (from $CRISPER_WHISPER_CLONED_DIR/requirements.txt)..."
if [ -f "$CRISPER_WHISPER_CLONED_DIR/requirements.txt" ]; then
    pip install -r "$CRISPER_WHISPER_CLONED_DIR/requirements.txt"
else
    echo "Warning: $CRISPER_WHISPER_CLONED_DIR/requirements.txt not found. Skipping CrisperWhisper deps."
    echo "This might cause issues if CrisperWhisper has specific dependencies like a custom transformers fork."
fi

echo "Installing main backend dependencies (from ./requirements.txt)..."
# Ensure PyTorch for ARM is handled correctly (should be if Python env is ARM)
# If CrisperWhisper reqs include torch, pip will manage it. Otherwise:
# pip install torch torchvision torchaudio
pip install -r ./requirements.txt

echo "Starting backend FastAPI server on http://localhost:8000 ..."
echo "Press Ctrl+C to stop the server."
# The processing_service.py is configured to find CrisperWhisper_cloned relative to backend/app
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Deactivation will happen when the script is stopped (Ctrl+C) and terminal session ends or a new command is run.
# If you want an explicit deactivate after stopping uvicorn, you'd need to trap SIGINT.
# For simplicity, this script ends when uvicorn ends.
echo "Backend server stopped."
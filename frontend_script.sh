#!/bin/bash
# frontend_script.sh - Sets up environment and runs the frontend Streamlit application.

echo "--- Starting Frontend Setup and Application ---"
set -e

# Navigate to the frontend directory from the project root
PROJECT_ROOT_DIR=$(pwd)
FRONTEND_DIR="$PROJECT_ROOT_DIR/frontend"

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "Error: Frontend directory not found at $FRONTEND_DIR"
    echo "Please run this script from the root of the 'pronunciation_app' project."
    exit 1
fi

cd "$FRONTEND_DIR"
echo "Changed directory to $(pwd)"

VENV_FRONTEND_DIR="./venv_frontend" # Relative to frontend directory

if [ ! -d "$VENV_FRONTEND_DIR" ]; then
    echo "Creating virtual environment for frontend at $VENV_FRONTEND_DIR..."
    python3 -m venv "$VENV_FRONTEND_DIR"
else
    echo "Virtual environment for frontend already exists at $VENV_FRONTEND_DIR."
fi

echo "Activating frontend virtual environment..."
source "$VENV_FRONTEND_DIR/bin/activate"

echo "Installing frontend dependencies (from ./requirements.txt)..."
pip install -r ./requirements.txt

echo "Starting frontend Streamlit application..."
echo "Streamlit will typically open in your browser at http://localhost:8501"
echo "Press Ctrl+C in this terminal to stop the application."
streamlit run app_streamlit.py

# Deactivation will happen when the script is stopped.
echo "Frontend application stopped."
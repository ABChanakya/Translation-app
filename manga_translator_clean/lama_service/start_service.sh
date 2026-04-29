#!/bin/bash
# lama_service/start_service.sh

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the venv directory
VENV_DIR="$DIR/venv_lama"

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating new virtual environment for LaMa service..."
    python3.12 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Check if dependencies are installed, if not, install them
# We check for a marker file to avoid running pip install every time
if [ ! -f "$VENV_DIR/deps_installed.marker" ]; then
    echo "📦 Installing dependencies for LaMa service..."
    pip install --upgrade pip
    pip install -r "$DIR/requirements.txt"
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully."
        touch "$VENV_DIR/deps_installed.marker"
    else
        echo "❌ Failed to install dependencies. Please check requirements.txt and run the script again."
        exit 1
    fi
fi

# Run from the service directory so gunicorn can find app.py
cd "$DIR"

if lsof -i TCP:5001 >/dev/null 2>&1; then
    echo "⚠️  Port 5001 is already in use. Reusing the existing LaMa service instead of starting another instance."
    exit 0
fi

# Use GPU by default — the pipeline unloads Ollama before inpainting starts,
# so there is no conflict. Override with: LAMA_DEVICE=cpu bash start_service.sh
export LAMA_DEVICE="${LAMA_DEVICE:-auto}"
echo "🖥️  LAMA_DEVICE=$LAMA_DEVICE"

# Run the Flask app using Gunicorn for better stability
echo "🚀 Starting LaMa inpainting service on http://0.0.0.0:5001..."
gunicorn --workers 1 --bind 0.0.0.0:5001 --timeout 300 app:app

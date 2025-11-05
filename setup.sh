#!/bin/bash

# MediaPipe Model Maker Environment Setup Script
# For macOS (Apple Silicon) with Python 3.11

echo "=========================================="
echo "MediaPipe Model Maker - Environment Setup"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 is not installed."
    echo "   Please install Python 3.11 first:"
    echo "   brew install python@3.11"
    exit 1
fi

echo "✅ Python 3.11 found"
python3.11 --version

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3.11 -m venv .venv311

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv311/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
echo "   This may take several minutes..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import tensorflow as tf
from mediapipe_model_maker import object_detector
print(f'✅ TensorFlow version: {tf.__version__}')
print('✅ MediaPipe Model Maker imported successfully')
print('')
print('🎉 Setup complete!')
print('')
print('To activate the environment in the future, run:')
print('  source .venv311/bin/activate')
"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="

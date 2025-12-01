#!/bin/bash

# --- Project Initialization Script (run.sh) ---

# 1. Setup Python Environment
echo "Setting up Python virtual environment..."
# Create a virtual environment named '.venv' if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate
echo "Virtual environment activated."

# 2. Install Dependencies
echo "Installing required Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the project in editable mode
# The pyproject.toml defines the 'AI' command entry point.
echo "Installing project in editable mode (-e .)..."
pip install -e .

# 4. Create necessary project directories
echo "Ensuring required data and model directories exist..."

# Directories needed for the data pipeline (github does not store empty folders)
mkdir -p data                  # For raw data
mkdir -p data/initial          # For cleaned data, to be transformed to engineered
mkdir -p data/intermediate     # For engineered data (Input for 'AI model')
mkdir -p data/metrics          # For saving JSON metrics (Output of 'AI model')
mkdir -p data/models           # For saving trained models (Output of 'AI model --save')

echo "Initialization complete. The virtual environment setup is transient."
echo "To use the new 'AI' CLI command, please run the following:"
echo ""
echo "source .venv/bin/activate  (to activate the environment)"
echo ""
echo "Available CLI commands: preprocess, engineer, model"
echo "Run using: AI <command>"
echo "Example: AI preprocess"
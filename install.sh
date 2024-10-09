#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Install the package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Uninstall uvloop if it's installed
echo "Uninstalling uvloop..."
pip uninstall uvloop -y

# Install mathgenerator
echo "Installing mathgenerator..."
pip install git+https://github.com/lukew3/mathgenerator.git

echo "Setup complete!"

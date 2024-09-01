#!/bin/bash

# Uninstalling mathgenerator
pip uninstall mathgenerator -y

# Install the package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Uninstall uvloop if it's installed
echo "Uninstalling uvloop..."
pip uninstall uvloop -y

# Install Bittensor version 6.9.3
echo "Installing Bittensor version 6.9.3..."
pip install bittensor==6.9.3

# Install the 6.9.4 patch from the GitHub repository
echo "Installing Bittensor 6.9.4 patch..."
pip install git+https://github.com/opentensor/bittensor.git@release/6.9.4

# Install mathgenerator
echo "Installing mathgenerator..."
pip install git+https://github.com/lukew3/mathgenerator.git

echo "Setup complete!"

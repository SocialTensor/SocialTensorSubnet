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

# add use torch to env echo "USE_TORCH=1" >> .env need to go down 1 line or else it will be at the end of the file
sed -i '$ d' .env
echo "USE_TORCH=1" >> .env

# check if use_torch is set
if grep -q "USE_TORCH=1" .env; then
    echo "Successfully set USE_TORCH=1"
else
    echo "Failed to set USE_TORCH=1"
    echo "Please set USE_TORCH=1 manually in the .env file"
fi

echo "Setup complete!"

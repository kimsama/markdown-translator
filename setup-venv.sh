#!/bin/bash

echo "Setting up virtual environment for AI Translator..."
echo ""

# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

echo ""
echo "Virtual environment 'venv' has been created and requirements installed."
echo ""
echo "To activate the virtual environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To use the translator, first activate the environment, then run:"
echo "    python translator.py -f path/to/file.md"
echo "    python translator.py -d path/to/directory"
echo ""

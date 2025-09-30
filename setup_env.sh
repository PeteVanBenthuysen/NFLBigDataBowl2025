#!/bin/bash
echo "========================================"
echo "NFL Big Data Bowl 2026 - Environment Setup"
echo "========================================"

echo
echo "Creating virtual environment..."
python3 -m venv nfl_env

echo
echo "Activating virtual environment..."
source nfl_env/bin/activate

echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=nfl_env --display-name="NFL Big Data Bowl"

echo
echo "Verifying installation..."
python src/utils.py

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To activate environment: source nfl_env/bin/activate"
echo "To start Jupyter: jupyter notebook notebooks/"
echo "To deactivate: deactivate"
echo
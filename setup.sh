#!/bin/bash
# Quick setup script for HAF implementation

echo "=========================================="
echo "HAF Implementation Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check installation
echo ""
echo "Verifying installation..."
python -c "import numpy; import scipy; import matplotlib; print('✓ Core dependencies installed')"

# Optional: Install yfinance
echo ""
read -p "Install yfinance for real data experiments? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install yfinance
    echo "✓ yfinance installed"
fi

# Run quick test
echo ""
echo "Running quick test..."
python -c "from haf_core import HausdorffAdaptiveFilter; import numpy as np; haf = HausdorffAdaptiveFilter(n_assets=1); haf.update(np.array([0.001])); print('✓ HAF core working')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run demo:        python demo.py"
echo "  2. Run tests:       pytest tests/ -v"
echo "  3. Reproduce paper: python reproduce_paper.py"
echo ""
echo "See README.md for more information."

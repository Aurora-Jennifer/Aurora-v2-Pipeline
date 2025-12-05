#!/bin/bash

# Build script for C++ trading kernels
set -e

echo "Building IBKR Trading Engine C++ Kernels..."

# Check dependencies
echo "Checking dependencies..."

# Check for pybind11
python3 -c "import pybind11; print(f'pybind11 version: {pybind11.__version__}')" || {
    echo "Installing pybind11..."
    pip3 install pybind11
}

# Check for numpy
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')" || {
    echo "Installing numpy..."
    pip3 install numpy
}

# Check for Eigen
if [ ! -d "/usr/include/eigen3" ] && [ ! -d "/usr/local/include/eigen3" ]; then
    echo "Installing Eigen..."
    sudo apt-get update
    sudo apt-get install -y libeigen3-dev
fi

# Build the extension
echo "Building C++ extension..."
python3 setup.py build_ext --inplace

echo "Build complete!"
echo "You can now import ibkr_trading_engine_py in Python"

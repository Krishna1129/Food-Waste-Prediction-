#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt

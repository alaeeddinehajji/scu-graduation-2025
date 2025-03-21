# scu-graduation-2025

# Multimodal EMG-EEG Classification

You should have a clear README.md explaining:
Project overview/purpose
Data description and format
Installation/setup instructions
How to run the code
Results interpretation
Dependencies/requirements list

## Project Overview

This project implements a deep learning model for multimodal classification using EMG and EEG signals.

## Data

- EMG data: 8 channels of electromyography signals
- EEG data: 8 channels of electroencephalography signals
- Classes: 7 different gestures/movements
- Data format: CSV files with aligned EMG and EEG recordings

## Setup

2. Place data files in:

- data/processed/EMG-data.csv
- data/processed/EEG-data.csv

## Usage

Run the Jupyter notebook:

## Results

- Model achieves ~77% test accuracy
- F1 Score: 0.76
- Results saved in model_results.npy

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Jupyter

1. Install dependencies:

torch>=1.8.0
numpy>=1.19.2
pandas>=1.2.3
scikit-learn>=0.24.1
jupyter>=1.0.0
matplotlib>=3.3.4

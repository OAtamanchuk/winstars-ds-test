# Winstars AI DS Internship Test - Task 1: MNIST Classification with OOP

## Overview

This task implements three classifiers for the MNIST handwritten digit dataset using a clean OOP structure:

- **Random Forest** (scikit-learn) - classical ML baseline  
- **Feed-Forward Neural Network** (PyTorch) - simple MLP with dropout  
- **Convolutional Neural Network** (PyTorch) - CNN with conv layers, batch norm, and dropout  

All models implement the same `MnistClassifierInterface` with `train` and `predict` methods.  
They are wrapped in a single `MnistClassifier` class that accepts an `algorithm` parameter (`'rf'`, `'nn'`, or `'cnn'`) and provides a **unified API** - the same input/output structure for all models regardless of the underlying algorithm.

Key highlights:
- Pixel normalization inside each model  
- GPU support (if available) for NN and CNN  
- Validation split and early stopping for neural models  
- Learning curves, confusion matrices, and edge case handling in the demo

## Project Structure
```
first_task/   
├── models   
│   ├── __init__.py   
│   ├── cnn.py   
│   ├── interface.py   
│   ├── neural_network.py   
│   └── random_forest.py   
├── demo.ipynb             # Main demonstration notebook with examples and edge cases   
├── mnist_classifier.py    # Unified wrapper class   
├── README.md              # This file   
└── requirements.txt       # Dependencies   
```
## Setup Instructions

1. Clone the repository  
   ```bash
   git clone https://github.com/OAtamanchuk/winstars-ds-test.git
   cd winstars-ds-test/first_task

2. Install all required packages
   ```
    pip install -r requirements.txt
4. Launch the Jupyter notebook
   ```
    jupyter notebook demo.ipynb
   ```
   or use JupyterLab (if preferred):
   ```
    jupyter lab
   ```
The **demo.ipynb** notebook contains: 
- Dataset loading and visualization
- Training and evaluation of all three models
- Learning curves and confusion matrices
- Overfitting checks
- Edge case demonstrations

**Note:** The jupyter package is included in requirements.txt, so running the above commands should be sufficient. 

## Dependencies (requirements.txt)

Main packages used:
- torch
- numpy
- scikit-learn
- matplotlib
- seaborn
- pandas
- tensorflow (for mnist.load_data)
- jupyter

Install with:
```
pip install -r requirements.txt
```
## Main Results

- Random Forest: 96.11% test accuracy 
- Feed-Forward NN: 96.55%
- CNN: 99.01% (30k subset), 99.42% on full dataset

Overfitting is minimal thanks to dropout, batch normalization, and early stopping.
Edge cases are handled correctly (noisy inputs, small datasets, invalid shapes, empty batches, random data).

## Notes

- Random seeds are fixed for reproducibility
- The demo uses a 30k subset for faster execution, full-dataset results (for CNN) are included optionally

This solution fully meets the task requirements: OOP design with interface, unified wrapper, three distinct models, comprehensive Jupyter demo with edge cases, and clear explanations.

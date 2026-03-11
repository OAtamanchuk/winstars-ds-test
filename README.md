# Data Science Test Assignment

This repository contains the solution for the Data Science test assignment.
The project includes two independent tasks covering several areas of Data Science, including:

- Machine Learning
- Computer Vision
- Natural Language Processing

All code is written in **Python 3** and organized into separate folders according to the task requirements.

# Repository Structure

```
├── first_task
│   ├── README.md
│   ├── requirements.txt
│   ├── models/
│   ├── demo.ipinb
│   └── mnist_classifier.py
│
├── second_task
│   ├── README.md
│   ├── requirements.txt
│   ├── data/
│   ├── eda/
│   ├── images/
│   ├── models/
│   ├── ner/
│   ├── vision/
│   ├── pipeline.py
│   └── demo.ipynb
│
└── README.md
```

Each task folder contains:

- its **own README with detailed explanations**
- **requirements.txt** with required libraries
- **training and inference scripts**
- **Jupyter notebooks with demonstrations and analysis**

# Task 1 - MNIST Image Classification (OOP)

The goal of the first task is to implement an image classification system for the **MNIST dataset** using three different machine learning models.

Implemented models:

- Random Forest
- Feed-Forward Neural Network
- Convolutional Neural Network (CNN)

The implementation follows an **object-oriented design**:

- `MnistClassifierInterface` - abstract interface with `train()` and `predict()` methods
- Individual classes implementing the interface:
- 
  * `RandomForestMnistClassifier`
  * `NeuralNetworkMnistClassifier`
  * `CNNMnistClassifier`
- `MnistClassifier` - wrapper class that selects the algorithm (`rf`, `nn`, `cnn`) and provides a unified prediction interface.

Additional materials included:

- Jupyter Notebook with examples and demonstrations
- training and evaluation scripts

More details can be found in:

**`first_task/README.md`**

---

# Task 2 - Animal Recognition Pipeline (NER + Computer Vision)

The second task implements a **machine learning pipeline** combining **Natural Language Processing** and **Computer Vision**.

The pipeline determines whether a textual statement about an image is **true or false**.

Example:

Text:

```
"There is a cow in the picture."
```

Image:

```
(image containing a cow)
```

Output:

```
True
```

Pipeline workflow:

1. A **NER model** extracts animal names from the text.
2. An **image classification model** predicts the animal shown in the image.
3. The pipeline compares both outputs and returns a **boolean result**.

Components included in the solution:

### NER model

- Transformer-based model for **animal name extraction**
- Parameterized training and inference scripts

### Image classification model

- CNN model trained on an **animal image dataset with 10 classes**
- Parameterized training and inference scripts

### ML Pipeline

- Python script that:

  * accepts **text and image as input**
  * processes them through both models
  * returns **True / False**

Additional materials:

- Jupyter notebook with **dataset exploration (EDA)**
- Jupyter notebook demonstrating **pipeline usage and edge cases**

More details can be found in:

**`second_task/README.md`**

# How to Use This Repository

Each task can be set up and run **independently**.

Please refer to the README file inside each task folder for:

- dataset preparation
- model training
- running inference
- running the full pipeline

# Technologies Used

Main libraries used in the project include:

- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- torchvision
- NumPy
- pandas
- matplotlib

# Notes
- Example usage and edge cases are demonstrated in **Jupyter notebooks**.

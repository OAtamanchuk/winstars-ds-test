import numpy as np
from models.random_forest import RandomForestMnistClassifier
from models.neural_network import NeuralNetworkMnistClassifier
from models.cnn import CNNMnistClassifier


class MnistClassifier:
    """
    Wrapper class for MNIST classifiers.

    Provides a unified interface for different algorithms.
    """

    def __init__(self, algorithm: str, **kwargs):

        algorithm = algorithm.lower()

        if algorithm == "rf":
            self.model = RandomForestMnistClassifier()

        elif algorithm == "nn":
            self.model = NeuralNetworkMnistClassifier(**kwargs)

        elif algorithm == "cnn":
            self.model = CNNMnistClassifier(**kwargs)

        else:
            raise ValueError("Unsupported algorithm. Use 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train selected model.
        """
        self.model.train(X_train, y_train, X_val, y_val)

    def predict(self, X):
        """
        Predict labels.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be numpy.ndarray")
        if X.ndim != 3 or X.shape[1:] != (28, 28):
            raise ValueError(f"Expected images (N, 28, 28), got shape {X.shape}")
        return self.model.predict(X)

    def get_metrics(self):
        return self.model.get_metrics()
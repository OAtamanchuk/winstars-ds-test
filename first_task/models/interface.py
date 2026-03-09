from abc import ABC, abstractmethod
import numpy as np


class MnistClassifierInterface(ABC):
    """
    Abstract interface for MNIST classifiers.

    All models must implement the same API so they can be used
    interchangeably through the MnistClassifier wrapper class.
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict digit classes for input images.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Return training metrics such as accuracy or loss history.
        """
        pass
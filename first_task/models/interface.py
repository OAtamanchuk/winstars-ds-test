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

        Parameters
        ----------
        X_train : np.ndarray
            Training images (N, 28, 28)

        y_train : np.ndarray
            Training labels

        X_val : np.ndarray
            Optional validation images

        y_val : np.ndarray
            Optional validation labels
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict digit classes for input images.

        Parameters
        ----------
        X : np.ndarray
            Images (N, 28, 28)

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Return training metrics such as accuracy or loss history.
        """
        pass
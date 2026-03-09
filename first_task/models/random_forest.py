import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST digits.

    Random Forest is a classical machine learning model that works
    with flattened feature vectors instead of images.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        self.train_accuracy = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:

        # Normalize images
        X_train = X_train.astype(np.float32) / 255.0

        # Flatten images (28x28 to 784)
        X_train_flat = X_train.reshape(len(X_train), -1)

        # Fit model
        self.model.fit(X_train_flat, y_train)

        # Evaluate training accuracy (used for overfitting detection)
        train_pred = self.model.predict(X_train_flat)
        self.train_accuracy = accuracy_score(y_train, train_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])

        X = X.astype(np.float32) / 255.0
        X_flat = X.reshape(len(X), -1)

        return self.model.predict(X_flat)

    def get_metrics(self):
        return {
            "train_accuracy": self.train_accuracy
        }
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)  # Import Classes from Sklearn.
from igann import IGANN  # Import IGANN class
import numpy as np


class IGANNClassifier(IGANN, BaseEstimator, ClassifierMixin):
    """
    IGANNClassifier integrates the IGANN model with scikit-learn's BaseEstimator and ClassifierMixin.

    This class serves as a wrapper to provide compatibility with scikit-learn's API for classification tasks.
    It primarily ensures that the IGANN model can be used with scikit-learn's utilities and conventions (e.g. GridSearchCV).

    For detailed documentation on the model's functionality, parameters, and methods, refer to the original IGANN class.

    Parameters:
    - **params (dict): Parameters to be passed to the IGANN constructor. 'task' parameter is set to 'classification'.
    """

    def __init__(self, **params):
        params.pop("task", None)  # Remove 'task' if it exists in params
        super(IGANNClassifier, self).__init__(task="classification", **params)
        # Additional initialization if needed

    def fit(self, X, y):
        """
        Fit the IGANN model to the provided data for classification tasks.

        Parameters:
        - X (array-like): Feature dataset.
        - y (array-like): Target labels.

        Returns:
        - self: Returns an instance of self.
        """
        # Fit the model using the parent class's fit method
        super(IGANNClassifier, self).fit(X, y)

        # Set the classes_ attribute
        self.classes_ = np.unique(y)
        return self


class IGANNRegressor(IGANN, BaseEstimator, RegressorMixin):
    """
    IGANNRegressor integrates the IGANN model with scikit-learn's BaseEstimator and RegressorMixin.

    This class serves as a wrapper to provide compatibility with scikit-learn's API for regression tasks.
    It primarily ensures that the IGANN model can be used with scikit-learn's utilities and conventions (e.g. GridSearchCV).

    For detailed documentation on the model's functionality, parameters, and methods, refer to the original IGANN class.

    Parameters:
    - **params (dict): Parameters to be passed to the IGANN constructor. 'task' parameter is set to 'regression'.
    """

    def __init__(self, **params):
        params.pop("task", None)  # Remove 'task' if it exists in params
        super(IGANNRegressor, self).__init__(task="regression", **params)

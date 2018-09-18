import numpy as np


class Perceptron(object):
    """Perceptron classifier"""

    def __init__(self, eta: float = 0.01, n_iter: int = 10):
        """
        :param eta: Step of training (between 0.0 and 1.0).
        :param n_iter:  Iterations for training based on the training sample.
        :var _weights: Weights after training.
        :var _errors:  Error on every epoch.
        :var _activation_functions: Activation functions
        """
        self.eta = eta
        self.n_iter = n_iter
        self._weights = None
        self._errors = None
        self._activation_functions = {
            'threshold': lambda x: np.where(x >= 0.0, 1, 0),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'hyperbolic_tangent': lambda x: np.tanh(x),
            'relu': lambda x: np.where(x >= 0.0, x, 0)
        }

    def fit(self, matrix_x: list, y: list) -> object:
        """
        Fit training data.
        :param matrix_x: The training vector is the input, where n_samples is the number of examples
         and n_features is the number of properties.
        :type matrix_x: {array-like}, shape = [n_samples, n_features]
        :param y: The output target value.
        :type y: array-like, shape = [n_samples]
        :return: self
        :rtype: object
        """
        self._weights = np.zeros(1 + matrix_x.shape[1])
        self._errors = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(matrix_x, y):
                update = self.eta * (target - self.predict(xi))
                self._weights[1:] += update * xi
                self._weights[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
        return self

    def net_input(self, matrix_x: list):
        """Calculate net input"""
        return np.dot(matrix_x, self._weights[1:]) + self._weights[0]

    def predict(self, matrix_x: list, func_name: str='threshold'):
        """Return class label after unit step"""
        func = self._activation_functions.get(func_name, self._activation_functions['threshold'])
        return func(self.net_input(matrix_x))

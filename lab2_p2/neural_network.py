import numpy as np


class NeuralNetwork(object):
    """NeuralNetwork using multi layer Perceptron classifier"""

    def __init__(self, arch, matrix_A, matrix_Y, num_iters=250, fit_step=0.5):
        """
        :param arch: Architecture of network layers (for ex 2-4-4-1).
        :param num_iters:  Iterations for training based on the training sample.
        """
        self.arch = arch
        self._n_arch = len(self.arch)
        self._weights = []
        self._biases = []
        self.fill_weights()
        self.fill_biases()
        self.output = []
        for o in range(self._n_arch):
            self.output.append(np.zeros([self.arch[o]]))
        self.grad_errors = []
        for ge in range(1, self._n_arch):
            self.grad_errors.append(np.zeros(self.arch[ge]))
        self.matrix_A = matrix_A
        self.matrix_Y = matrix_Y
        self.num_iters = num_iters
        self.fit_step = fit_step
        
        self._activation_functions = {
            'threshold': {
                'function': lambda x: np.where(x >= 0.0, 1, 0),
                'derivative': lambda x: np.where(x >= 0.0, 1, 0)
            },
            'sigmoid': {
                'function': lambda x: 1 / (1 + np.exp(-x)),
                'derivative': lambda x: 1 / (1 + np.exp(-x))
            },
            'linear': {
                'function': lambda x: x,
                'derivative': lambda x: np.ones(x.shape)
            },
            'hyperbolic_tangent': {
                'function': lambda x: np.tanh(x),
                'derivative': lambda x: 1 - np.square(x)
            },
            'relu': {
                'function': lambda x: np.where(x >= 0.0, x, 0),
                'derivative': lambda x: np.where(x >= 0.0, x, 0)
            }
        }

        self.act_funcs = []
        self.der_funcs = []
        self.fill_activation_functions()

    @property
    def weights(self):
        return self._weights

    @property
    def n_arch(self):
        return self._n_arch

    def fill_biases(self):
        for b in range(1, self._n_arch):
            self._biases.append(np.random.randn(self.arch[b]) * 0.1)

    def fill_weights(self):
        for w in range(self._n_arch - 1):
            self._weights.append(np.random.randn(self.arch[w], self.arch[w + 1]) * 0.1)

    def fill_activation_functions(self):
        for lap in range(self._n_arch - 1):
            func = self._activation_functions['hyperbolic_tangent']
            self.act_funcs.append(func['function'])
            self.der_funcs.append(func['derivative'])
        func = self._activation_functions['linear']
        self.act_funcs.append(func['function'])
        self.der_funcs.append(func['derivative'])

    def fit(self):
        # Цикл для каждой итерации
        for c in range(self.num_iters):
            # Цикл по всем входным векторам
            for i in range(len(self.matrix_A)):
                # Целевой вектор
                t = self.matrix_Y[i, :]
                # Шаг вперед
                self.output[0] = self.matrix_A[i, :]
                for j in range(self._n_arch - 1):
                    self.output[j + 1] = self.act_funcs[j](np.dot(self.output[j], self._weights[j]) + self._biases[j])
                print('Выход:' + str(self.output[-1]))
                # Вычисление градиента ошибки выходного нейрона
                self.grad_errors[-1] = np.multiply((t - self.output[-1]), self.der_funcs[-1](self.output[-1]))
                # Вычисление градиента ошибки промежуточных нейронов
                for j in range(self._n_arch - 2, 0, -1):
                    self.grad_errors[j - 1] = np.multiply(np.dot(self.grad_errors[j], self._weights[j].T),
                                                          self.der_funcs[j](self.output[j]))
                # Обновление веса и смещения
                for j in range(self._n_arch - 1):
                    self._weights[j] = self._weights[j] + self.fit_step * np.outer(self.output[j], self.grad_errors[j])
                    self._biases[j] = self._biases[j] + self.fit_step * self.grad_errors[j]


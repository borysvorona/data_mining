import numpy as np

from neural_network import NeuralNetwork

if __name__ == "__main__":

    # Architecture 2-4-4-1
    arch = np.array([2, 4, 4, 1])
    # Обучающий Входной массив (каждый ряд - наблюдение)
    matrix_A = np.matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    # Целевые значения (каждый ряд - наблюдение)
    matrix_Y = np.matrix([[-0.5], [0.5], [0.5], [-0.5]])

    nn = NeuralNetwork(arch=arch, matrix_A=matrix_A, matrix_Y=matrix_Y, num_iters=250, fit_step=0.5)
    nn.fit()

    print('\nВеса после обучения:')
    # Выывод финальных весов сети
    for i in range(nn.n_arch - 1):
        print('Слой {0}:\n{1}\n'.format(str(i + 1), str(nn.weights[i])))

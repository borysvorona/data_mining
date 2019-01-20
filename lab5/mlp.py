from keras.layers import Dense
from keras.models import Sequential


class MultilayerPerceptron(object):
    def __init__(self, x, y):
        """
        Initialize object for Multilayer perceptron Neural Network
        :param x: Inputs for fitting Neuron Neural
        :param y: Outputs for fitting Neuron Neural
        """
        self.x = x
        self.y = y
        self.model = Sequential()
        self.model.add(Dense(3, input_dim=3, activation='softmax'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.add(Dense(1, activation='softmax'))

    def run(self):
        self.compile()
        self.fit()
        self.check()

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def fit(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def evaluate(self):
        return self.model.evaluate(self.x, self.y)

    def predict(self, x):
        return self.model.predict(x)

    def check(self):
        scores = self.evaluate()
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

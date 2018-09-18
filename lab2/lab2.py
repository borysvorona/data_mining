import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron

if __name__ == "__main__":
    df = pd.read_csv('./data/train.data', header=None)
    y = df.iloc[0:8, 3].values
    yy = y
    X = df.iloc[0:8, [0, 1, 2]].values

    ppn = Perceptron(eta=0.01, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn._errors) + 1), ppn._errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    print(ppn.predict(X))

    error_last = yy - ppn.predict(X)
    print(error_last)

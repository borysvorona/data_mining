import math
import pandas as pd

from mlp import MultilayerPerceptron


if __name__ == "__main__":
    file_path = './data/train.csv'
    data_frame = pd.read_csv(file_path, sep=',')
    # data preparation
    data_frame['Sex'] = data_frame['Sex'].map({'female': 1, 'male': 0})
    data_frame['Age'] = pd.to_numeric(data_frame['Age'], errors='coerce')
    data_frame['Age'] = data_frame['Age'].map(lambda a: int(a) >= 18 if not math.isnan(a) else 0)

    x = data_frame[['Age', 'Sex', 'Pclass']].values
    y = data_frame['Survived'].values

    # Neural network creation and fitting
    mlp = MultilayerPerceptron(x=x, y=y)
    mlp.run()

    result = mlp.predict(x)
    print(result)

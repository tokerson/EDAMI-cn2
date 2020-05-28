import os
import pandas as pd
from src.cn2 import CN2

path = os.path.dirname(os.getcwd())


def load_data(filename):
    colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'result']
    return pd.read_csv(filename, names=colnames, header=None)


if __name__ == "__main__":
    training_set = load_data(os.path.join(path, 'data', 'training.csv'))
    test_set = load_data(os.path.join(path, 'data', 'test.csv'))
    cn2 = CN2(training_set, test_set)
    cn2.learn()

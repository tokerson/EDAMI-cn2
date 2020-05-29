import os
import pandas as pd
from src.cn2 import CN2

path = os.path.dirname(os.getcwd())

colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'result']

def load_data(filename):
    return pd.read_csv(filename, names=colnames, skiprows=1, header=None)


if __name__ == "__main__":
    training_set = load_data(os.path.join(path, 'data', 'training.csv'))
    test_set = load_data(os.path.join(path, 'data', 'test.csv'))
    cn2 = CN2(training_set, colnames)
    rule_list = cn2.learn()
    print(rule_list)

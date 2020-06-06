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
    training_set = load_data(os.path.join(path, 'data', 'adult','training.csv'))
    test_set = load_data(os.path.join(path, 'data', 'adult', 'test.csv'))
    cn2 = CN2(training_set, colnames, colnames[-1])
    rule_list = cn2.learn()
    test_results = cn2.test(test_set, rule_list)
    print(rule_list)
    print(test_results)

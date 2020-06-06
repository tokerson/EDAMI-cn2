import os
import pandas as pd
from src.cn2 import CN2

path = os.path.dirname(os.getcwd())

adult_colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'result']

car_colnames = ['buying-price','maint-price','doors','persons','lug_boot','safety','class']

def load_data(filename, colnames):
    return pd.read_csv(filename, names=colnames, skiprows=1, header=None)


if __name__ == "__main__":
    training_set = load_data(os.path.join(path, 'data', 'cars','training.csv'),car_colnames)
    test_set = load_data(os.path.join(path, 'data', 'cars', 'test.csv'), car_colnames)
    cn2 = CN2(training_set, car_colnames, car_colnames[-1], min_significance=0.5)
    rule_list = cn2.learn()
    print(len(rule_list))
    test_results = cn2.test(test_set, rule_list)
    print(rule_list)
    print(test_results)

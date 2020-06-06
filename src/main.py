import os
import pandas as pd
import time
from src.cn2 import CN2

path = os.path.dirname(os.getcwd())

adult_colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                  'result']

car_colnames = ['buying-price', 'maint-price', 'doors', 'persons', 'lug_boot', 'safety', 'class']

nursery_colnames = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']


def load_data(filename, colnames):
    return pd.read_csv(filename, names=colnames, skiprows=1, header=None)


def run_cn2_on_cars_dataset(min_significance=0.5, star_max_size=5):
    print('-----------------')
    print('CARS')
    print('-----------------')
    training_set = load_data(os.path.join(path, 'data', 'cars', 'training.csv'), car_colnames)
    test_set = load_data(os.path.join(path, 'data', 'cars', 'test.csv'), car_colnames)
    cn2 = CN2(training_set, car_colnames, car_colnames[-1], min_significance=min_significance,
              star_max_size=star_max_size)
    rule_list = run_cn2(cn2)
    # test_results: (correct, incorrect, uncovered, total, rules_accuracy_dict)
    test_results = cn2.test(test_set, rule_list)
    print(test_results)

def run_cn2_on_nursery_dataset(min_significance=0.5, star_max_size=5):
    print('-----------------')
    print('NURSERY')
    print('-----------------')
    training_set = load_data(os.path.join(path, 'data', 'nursery', 'training.csv'), nursery_colnames)
    test_set = load_data(os.path.join(path, 'data', 'nursery', 'test.csv'), nursery_colnames)
    cn2 = CN2(training_set, nursery_colnames, nursery_colnames[-1], min_significance=min_significance,
              star_max_size=star_max_size)
    rule_list = run_cn2(cn2)
    # test_results: (correct, incorrect, uncovered, total, rules_accuracy_dict)
    test_results = cn2.test(test_set, rule_list)
    print(test_results)

def run_cn2_on_adults_dataset(min_significance=0.5, star_max_size=5):
    print('-----------------')
    print('ADULTS')
    print('-----------------')
    training_set = load_data(os.path.join(path, 'data', 'adult', 'training.csv'), adult_colnames)
    test_set = load_data(os.path.join(path, 'data', 'adult', 'test.csv'), adult_colnames)
    cn2 = CN2(training_set, adult_colnames, adult_colnames[-1], min_significance=min_significance,
              star_max_size=star_max_size)
    rule_list = run_cn2(cn2)
    # test_results: (correct, incorrect, uncovered, total, rules_accuracy_dict)
    test_results = cn2.test(test_set, rule_list)
    print(test_results)

def run_cn2(cn2):
    learning_start = time.time()
    rule_list = cn2.learn()
    learning_end = time.time()

    print("Learning time: {} seconds".format(learning_end - learning_start))
    print("Number of rules: {}".format(len(rule_list)))
    return rule_list

if __name__ == "__main__":
    # run_cn2_on_cars_dataset()
    run_cn2_on_nursery_dataset()
    # run_cn2_on_adults_dataset(min_significance=0.8)

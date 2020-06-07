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
    save_rules_to_file(os.path.join(path, 'results',
                                    "results_star={},significance={}".format(cn2.star_max_size, cn2.min_significance)), rule_list)
    return rule_list


def save_rules_to_file(filename, rules):
    f = open(filename, "w")
    for rule in rules:
        f.write(rule[0])
        f.write('\n')
    f.close()


if __name__ == "__main__":
    loop = True
    while loop:
        print("---------CN2--------------")
        print("1. Run algorithm for adult data")
        print("2. Run algorithm for car data")
        print("3. Run algorithm for nursery data")
        inp = input("Choose menu option, any other key leaves the program:")
        user_input = int(inp)
        if user_input > 3 or user_input < 1:
            loop = False
            break
        max_star_size = int(input("Insert max size of the star [int]:"))
        min_significance = float(input("Insert min significance [float, dot separated]:"))
        if user_input == 1:
            run_cn2_on_adults_dataset(min_significance=min_significance, star_max_size=max_star_size)
        elif user_input == 2:
            run_cn2_on_cars_dataset(min_significance=min_significance, star_max_size=max_star_size)
        elif user_input == 3:
            run_cn2_on_nursery_dataset(min_significance=min_significance, star_max_size=max_star_size)

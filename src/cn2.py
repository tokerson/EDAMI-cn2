import numpy as np

class CN2:
    def __init__(self, training_set):
        self.training_set = training_set

    def learn(self):
        rule_list = np.empty(0)
        while (self.training_set.empty == False):
            best_condition_expression = self.find_best_condition_expression()
            if (best_condition_expression is not None):
                training_subset = self.get_examples_covered_by_expression(best_condition_expression)
                # drop items by indexes in training_set
                self.training_set.drop([])
                most_common_class = self.get_most_common_class_from_subset(training_subset)
                np.append(rule_list, "if {} then the class is {}".format(best_condition_expression, most_common_class))

        return rule_list

    def find_best_condition_expression(self):
        pass

    def get_examples_covered_by_expression(self, best_condition_expression):
        pass

    def get_most_common_class_from_subset(self, training_subset):
        pass

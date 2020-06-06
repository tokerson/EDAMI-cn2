import numpy as np


class CN2:
    def __init__(self, training_set, attributes, class_col_name, min_significance=0.8, star_max_size=5):
        self.training_set = training_set
        self.attributes = attributes
        self.class_col_name = class_col_name
        self.classified_results = training_set[self.class_col_name].to_numpy()
        self.selectors = self.get_selectors(attributes)
        self.min_significance = min_significance
        self.star_max_size = star_max_size
        self.E = training_set.copy()
        self.update_train_probs()

    def get_selectors(self, attributes):
        _attributes = attributes[0:-1]  # remove last column from selector attributes
        selectors = []
        for attribute in _attributes:
            self.possible_values = self.training_set[attribute].unique()
            for value in self.possible_values:
                selectors.append((attribute, value))
        return selectors

    def learn(self):
        rule_list = []
        while (self.E.empty == False):
            best_complex = self.find_best_condition_expression()
            if (best_complex is not None):
                training_subset = self.get_examples_covered_by_complex(best_complex)
                # drop items by indexes in training_set
                self.E = self.E.drop(training_subset.index)
                self.update_train_probs()
                most_common_class = self.get_most_common_class_from_subset(training_subset)
                rule_list.append(("if {} then the class is {}".format(best_complex, most_common_class), best_complex,
                                  most_common_class))
            else:
                break

        return rule_list

    def find_best_condition_expression(self):
        star = []
        best_complex = None
        best_entropy = float('inf')
        best_significance = 0.0

        while True:
            new_star = self.set_new_star(star)
            complex_entropies = {}
            for index, complex in enumerate(new_star):
                significance, entropy = self.get_significance_and_entropy(complex)
                if significance > self.min_significance:
                    if entropy == 0.0: return complex.copy()
                    complex_entropies[index] = entropy
                    if entropy < best_entropy:
                        best_complex = complex.copy()
                        best_entropy = entropy
                        best_significance = significance

            # remove the worst complexes
            best_complex_indexes = sorted(complex_entropies.items(), key=lambda item: item[1])[
                                   0:self.star_max_size]

            star = [new_star[x[0]] for x in best_complex_indexes]
            if len(star) == 0 or best_significance < self.min_significance:
                break

        return best_complex

    def get_examples_covered_by_complex(self, best_complex):
        values = dict()
        [values[t[0]].append(t[1]) if t[0] in list(values.keys())
         else values.update({t[0]: [t[1]]}) for t in best_complex]

        for attribute in self.attributes:
            if attribute not in values:
                values[attribute] = self.training_set[attribute].unique()

        covered_examples = self.E[self.E.isin(values).all(axis=1)]
        return covered_examples

    def get_most_common_class_from_subset(self, training_subset):
        classes = self.training_set.loc[training_subset.index, [self.class_col_name]]
        return classes.iloc[:, 0].value_counts(sort=True).index[0]

    def set_new_star(self, star):
        new_star = []
        if len(star) > 0:
            for complex in star:
                for selector in self.selectors:
                    new_complex = self.get_new_complex(complex, selector)
                    if new_complex is not None:
                        new_star.append(new_complex)
        else:
            for selector in self.selectors:
                complex = [selector]
                new_star.append(complex)

        return new_star

    def get_new_complex(self, complex, selector):
        for _selector in complex:
            if selector[0] == _selector[0]:
                return None  # it is duplicate

        new_complex = complex.copy()
        new_complex.append(selector)
        return new_complex

    def get_significance_and_entropy(self, complex):
        covered_examples = self.get_examples_covered_by_complex(complex)
        classes_covered_by_complex_probs = self.get_covered_classes_probabilities(covered_examples)

        significance = self.calculate_significance(classes_covered_by_complex_probs)
        entropy = self.calculate_entropy(classes_covered_by_complex_probs)

        return (significance, entropy)

    def get_covered_classes_probabilities(self, covered_examples):
        classes = self.training_set.loc[covered_examples.index, [self.class_col_name]]
        covered_classes_counts = classes.iloc[:, 0].value_counts()
        return covered_classes_counts.divide(len(classes))

    def calculate_significance(self, covered_classes_probs):
        return covered_classes_probs.multiply(
            np.log(covered_classes_probs.divide(self.train_probs))).sum() * 2

    def calculate_entropy(self, covered_classes_probs):
        log2 = np.log2(covered_classes_probs)
        plog2p = covered_classes_probs.multiply(log2)
        return plog2p.sum() * -1

    def test(self, test_set, rules):
        test_data = test_set.iloc[:, :-1]
        total = len(test_data.index)
        correct = 0
        incorrect = 0

        # classification test
        for test_example in test_set.iterrows():
            for rule in rules:
                if self.is_test_example_covered_by_rule(test_example[1][:-1], rule[1]):
                    if rule[2] == test_example[1][self.class_col_name]:
                        correct += 1
                    else:
                        incorrect += 1
                    break

        # rules test
        rules_accuracy = dict()
        for rule in rules:
            rules_accuracy[rule[0]] = (0.0, 0 ,0)
            rule_correct = 0
            rule_incorrect = 0
            for test_example in test_set.iterrows():
                if self.is_test_example_covered_by_rule(test_example[1][:-1], rule[1]):
                    if rule[2] == test_example[1][self.class_col_name]:
                        rule_correct += 1
                    else:
                        rule_incorrect += 1
            rules_accuracy[rule[0]] = (rule_correct / (rule_correct + rule_incorrect), rule_correct, rule_incorrect)

        not_covered = total - correct - incorrect
        return (correct, incorrect, not_covered, total, rules_accuracy)

    def is_test_example_covered_by_rule(self, test_example, rule):
        for selector in rule:
            if test_example[selector[0]] != selector[1]:
                return False

        return True

    def update_train_probs(self):
        train_classes = self.E.iloc[:, -1]
        train_num_instances = len(train_classes)
        train_counts = train_classes.value_counts()
        self.train_probs = train_counts.divide(train_num_instances)

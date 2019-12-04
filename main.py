from classes.rule_based_ga import RuleBasedGeneticAlgorithm
from classes.adaptive_rule_based_ga import AdaptiveRuleBasedGeneticAlgorithm as ARBGA
from classes.rule_based_fp_ga import RuleBasedFloatingPointGA
from modules.data import open_and_sanitize_data, initialise_data

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys


def main():
    dirname = os.path.dirname(__file__)
    sys.path.append(dirname)
    data_four(dirname)


def data_one(dirname):
    data_path = os.path.join(dirname, "data/dataset1.txt")
    data = open_and_sanitize_data(data_path)
    ga = RuleBasedGeneticAlgorithm(dataset=data, mutation_probability=0.0047,
                                   crossover_probability=0.9, population_size=100, rule_count=30)
    ga.evolve(epochs=5000)


def data_two(dirname):
    data_path = os.path.join(dirname, "data/dataset2.txt")
    data = open_and_sanitize_data(data_path)
    ga = ARBGA(dataset=data, mutation_probability=0.005,
               population_size=100, rule_count=30)
    ga.evolve(epoch=5000)


def data_three(dirname):
    data_path = os.path.join(dirname, "data/dataset3.txt")
    data = open_and_sanitize_data(data_path)
    training_set = data[0::2]
    test_set = data[1::2]
    ga = RuleBasedFloatingPointGA(dataset=training_set, test_data=test_set,
                                  mutation_probability=0.01, crossover_probability=0.9,
                                  population_size=100, rule_count=5)
    ga.evolve(epoch=2500)


def data_four(dirname):
    data_path = os.path.join(dirname, "data/dataset4.txt")
    data = open_and_sanitize_data(data_path)
    training_set = data[0::2]
    test_set = data[1::2]
    training_set = training_set + test_set[0::2]
    test_set = test_set[1::2]
    # training_set = data[0::2]
    # test_set = data[1::2]
    ga = RuleBasedFloatingPointGA(dataset=training_set, test_data=test_set,
                                  mutation_probability=0.008, crossover_probability=0.9,
                                  population_size=50, rule_count=5, seed_edge_cases=True)
    ga.evolve(epoch=10000)


def evaluate_rules(dirname):
    data_two = os.path.join(dirname, "data/dataset2.txt")
    rules = os.path.join(dirname, "observations/data2rules.txt")
    data_two = open_and_sanitize_data(data_two)
    data_two = initialise_data(data_two)[0]

    with open(rules, mode="r") as f:
        data = f.readlines()
        for i in range(len(data)):
            line = data[i]
            data[i] = line.strip("\n")

    data = [(d[:len(d) - 1], d[-1]) for d in data]

    unmatched = []
    for bd in data_two:
        fm = False
        for (x, y) in data:
            if(match(bd.feature, x) and bd.label == y):
                fm = True
                break
        if(fm == False):
            unmatched.append(bd)
    print(unmatched)


def match(data, rule):
    return all((a == b) or b == '#' for a, b in zip(data, rule))


main()

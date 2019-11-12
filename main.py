from classes.rule_based_ga import RuleBasedGeneticAlgorithm
from classes.adaptive_rule_based_ga import AdaptiveRuleBasedGeneticAlgorithm as ARBGA
from classes.rule_based_fp_ga import RuleBasedFloatingPointGA
from modules.data import open_and_sanitize_data

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys

def main():
    dirname = os.path.dirname(__file__)
    sys.path.append(dirname)
    data_three(dirname)

def data_one(dirname):
    #TODO: setup for good data 1.
    data_path = os.path.join(dirname, "data/dataset1.txt")
    data = open_and_sanitize_data(data_path)
    ga = RuleBasedGeneticAlgorithm(dataset=data, mutation_probability=0.0125, 
        crossover_probability=0.8, population_size=100, rule_count=5)
    ga.evolve(epochs=2000)

def data_two(dirname):
    data_path = os.path.join(dirname, "data/dataset1.txt")
    data = open_and_sanitize_data(data_path)
    ga = ARBGA(dataset=data, mutation_probability=0.005,
        population_size=60, rule_count=42)
    ga.evolve(epoch=5000)

def data_three(dirname):
    data_path = os.path.join(dirname, "data/dataset3.txt")
    data = open_and_sanitize_data(data_path)
    training_set = data[0::2]
    test_set = data[1::2]
    ga = RuleBasedFloatingPointGA(dataset=training_set,test_data=test_set,
        mutation_probability=0.02, crossover_probability=0.9, 
        population_size=120, rule_count=20)
    ga.evolve(epoch=2000)

main()
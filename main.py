from classes.rule_based_ga import RuleBasedGeneticAlgorithm
from classes.adaptive_rule_based_ga import AdaptiveRuleBasedGeneticAlgorithm as ARBGA
from modules.data import open_and_sanitize_data
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys

def main():
    dirname = os.path.dirname(__file__)
    sys.path.append(dirname)
    data_two(dirname)


def data_one(dirname):
    #TODO: setup for good data 1.
    data_path = os.path.join(dirname, "data/dataset2.txt")
    data = open_and_sanitize_data(data_path)
    ga = RuleBasedGeneticAlgorithm(dataset=data, mutation_probability=0.00208333333, 
        crossover_probability=0.9, population_size=110, rule_count=30)
    ga.evolve(epochs=1000)

def data_two(dirname):
    data_path = os.path.join(dirname, "data/dataset2.txt")
    data = open_and_sanitize_data(data_path)
    ga = ARBGA(dataset=data, mutation_probability=0.005,
        population_size=120, rule_count=42)
    ga.evolve(epoch=5000)

main()
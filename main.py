from Assignment.GeneticAlgorithm import RuleBasedGeneticAlgorithm
import csv

def main():

    data = open_and_sanitize_data("Assignment/dataset1.txt")
    ga = RuleBasedGeneticAlgorithm(dataset=data, mutation_probability=0.0125, 
        crossover_probability=0.9, population_size=100, rule_count=5)
    ga.evolve(epochs=400)

def open_and_sanitize_data(data_set_location):
    with open(data_set_location, mode="r") as f:
        data = f.readlines()
        for i in range (len(data)):
            line = data[i]
            data[i] = line.strip("\n")
        return data

main()
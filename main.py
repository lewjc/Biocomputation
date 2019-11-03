from Assignment.classes.RuleBasedGeneticAlgorithm import RuleBasedGeneticAlgorithm
import os

def main():
    
    dirn = os.path.dirname(__file__)
    data_path = os.path.join(dirn, "data/dataset2.txt")
    data = open_and_sanitize_data(data_path)
    ga = RuleBasedGeneticAlgorithm(dataset=data, mutation_probability=0.000125, 
        crossover_probability=0.85, population_size=80, rule_count=30)
    ga.evolve(epochs=1000)

def open_and_sanitize_data(data_set_location):
    with open(data_set_location, mode="r") as f:
        data = f.readlines()
        for i in range (len(data)):
            line = data[i]
            data[i] = line.strip("\n")
        return data

main()
from classes.individual import Individual
import random
from copy import deepcopy

class GeneticAlgorithmBase():

    def __init__(self, population_size=10, chromosome_size=10, 
        mutation_probability=0.01, crossover_probability=0.9):
        self._population_size = population_size
        self._chromosome_size = chromosome_size
        self._mutation_probabilty = mutation_probability
        self._crossover_probability = crossover_probability
        self._population = []
        self._generation = 0


    def _get_worst_individual(self, population, with_index=False):
        worst = population[0]
        idx = 0
        worst_idx = 0
        for individual in population:
            if (individual.fitness < worst.fitness):
                worst = individual
                worst_idx = idx
            idx +=1 

        worst = deepcopy(worst)
        return (worst, worst_idx) if with_index else worst  
    
    def _get_best_individual(self, population, with_index=False):
        best = population[0]
        idx = 0
        best_idx = 0
        for individual in population:
            if (individual.fitness > best.fitness):
                best = individual
                best_idx = idx
            idx +=1 
        
        best = deepcopy(best)
        return (best, best_idx) if with_index else best
    
    
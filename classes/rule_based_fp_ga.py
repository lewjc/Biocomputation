from classes.base_ga import GeneticAlgorithmBase
from modules.data import initialise_floating_point_data
from classes.individual import Individual
from classes.rule import Rule
from modules.selection import tournament_selection
from modules.crossover import one_point_crossover
from modules.mutation import floating_point_boundary_mutate
from copy import deepcopy
import random
import uuid

class RuleBasedFloatingPointGA(GeneticAlgorithmBase):
    
    def __init__(self, dataset, population_size=50, mutation_probability=0.001,
        crossover_probability=0.8, rule_count=10):
        
        super().__init__(population_size=population_size,
            mutation_probability=mutation_probability,
            crossover_probability=crossover_probability)

        self._rule_count = rule_count
        (self._data, self._feature_size,
            self._label_size) = initialise_floating_point_data(dataset)
        
        self._rule_size = self._feature_size + self._label_size
        self._chromosome_size = self._rule_size * self._rule_count 
        self._initialise_population()
        self._generation_info = []
        self._max_fitness = len(self._data)
        self.evolve()


    def _initialise_population(self):
        for i in range(0, self._population_size):
            chromosome = self._generate_chromosome()
            self._population.insert(i,
                Individual(uuid.uuid1().hex, chromosome, 0))


    def _generate_chromosome(self):
        chromosome = []
        while len(chromosome) < self._chromosome_size:
            for i in range(self._feature_size):
                gene = lambda : random.random() if random.randint(0, 1) else '#'
                bounds = list((gene(), gene()))
                if(any(isinstance(x, str) for x in bounds)):
                    chromosome.append((bounds[0], bounds[1]))
                else:
                    chromosome.append((min(bounds), max(bounds)))

            for j in range(self._label_size):
                chromosome.append(random.choice([0,1]))

        return chromosome


    def evolve(self, epoch=1000):
        while(self._generation < epoch):
            for individual in self._population:
                self._evaluate_fitness(individual)

            self._display_population_fitness(self._population)
            self._population = self._generate_offspring(self._population)
            best = self._get_best_individual(self._population)
            if(best.fitness == self._max_fitness):
                break
            self._reset_population_fitness(self._population)
            self._generation +=1 
    

    def _evaluate_fitness(self, individual):        
        rules = Rule.generate_rules_from_chromosome(individual.chromosome,
            self._feature_size, self._rule_size)

        for data in self._data:
            for boundary_rule in rules:
                if(self._does_rule_match(data.features, boundary_rule.feature)):
                    if(data.prediction == boundary_rule.label):
                        individual.fitness += 1
                    break

    def _does_rule_match(self, rule, data):
        return all((b if str(b) != '#' else 0) <= a <= (c if str(c) != '#' else 1) 
            for a, (b, c) in zip(rule, data))

    def _display_population_fitness(self, population):
        best_individual = super()._get_best_individual(population)
        average_fitness = round(self._get_average_fitness(population), 2)
        print("[Gen] {} [Average] {} [Best] {}".format(
            self._generation, average_fitness, best_individual.fitness))
        self._generation_info.append([self._generation, average_fitness, best_individual.fitness])
    

    def _generate_offspring(self, population):
        best_individual = self._get_best_individual(population)
        offspring = tournament_selection(deepcopy(population))
        offspring = one_point_crossover(offspring, self._crossover_probability,
            self._chromosome_size)

        for individual in offspring:
            floating_point_boundary_mutate(individual, 
                self._mutation_probabilty, self._rule_size)

        _, worst_idx = self._get_worst_individual(population=population,
            with_index=True)

        offspring[worst_idx] = best_individual
        return offspring

    def _get_average_fitness(self, population):
        fitness_total = 0
        for individual in population:
            fitness_total += individual.fitness
        
        return fitness_total / self._population_size
    
    def _reset_population_fitness(self, population):
        for individual in population:
            individual.fitness = 0  
from Assignment.GeneticAlgorithmBase import GeneticAlgorithmBase, Individual
from Assignment.Classes.rule import Rule
from Assignment.Modules.data import initialise_data, draw_graph
from Assignment.Modules.selection import tournament_selection
from Assignment.Modules.crossover import one_point_crossover, two_point_crossover
from Assignment.Modules.mutation import rule_based_mutate
from copy import deepcopy
import random


class RuleBasedGeneticAlgorithm(GeneticAlgorithmBase):

    def __init__(self, dataset, population_size=10, 
        mutation_probability=0.05, crossover_probability=1, rule_count=20):
        
        super().__init__(population_size=population_size, 
            mutation_probability=mutation_probability, 
            crossover_probability=crossover_probability)

        self._rule_count = rule_count
        self._data, self._feature_size, self._label_size = initialise_data(dataset)
        self._rule_size  = self._feature_size + self._label_size
        self._chromosome_size = self._rule_count * self._rule_size
        self._generation_info = []
        self._max_fitness = len(dataset)

        self.__initialise_population()        


    def __initialise_population(self):
        for i in range(self._population_size):
            chromosome = self.__generate_chromosome()
            individual = Individual(i, chromosome=chromosome,
                fitness=0) 
            self._population.insert(i, individual)
        self._generation = 1


    def __generate_chromosome(self):
        chromosome = []
        while len(chromosome) < self._chromosome_size:
            for i in range(self._feature_size):
                chromosome.append(random.choice([0,1,'#']))

            for j in range(self._label_size):
                chromosome.append(random.choice([0,1]))

        return chromosome
    

    def __evaluate_fitness(self, individual):        
        rules = Rule.generate_rules_from_chromosome(individual.chromosome, 
            self._feature_size, self._rule_size)

        for data in self._data:
            for rule in rules:
                if(self.__does_rule_match_data(rule.feature, data.feature)):
                    if(rule.label == int(data.label)):
                        individual.fitness += 1                
                    break
    

    def __does_rule_match_data(self, rule, data):        
        return all(a == str(b) or str(b) == '#' for a, b in zip(data, rule))


    def evolve(self, epochs=2000):    
        while(self._generation < epochs):
            for individual in self._population:
                self.__evaluate_fitness(individual)
            self.__display_population_fitness(self._population)
            self._population = self.__generate_offspring(self._population)
            best = self._get_best_individual(self._population)
            if(best.fitness == self._max_fitness):
                break
            self.__reset_population_fitness(self._population)
            self._generation +=1 
            
        
        rules = Rule.generate_rules_from_chromosome(best.chromosome, self._feature_size, 
            self._rule_size)

        print("[BEST RULE BASE]")
        for rule in rules:
            print("RULE: {} {}".format(rule.feature, rule.label))

        draw_graph(self._generation_info, epochs, self._crossover_probability,
            self._mutation_probabilty, self._rule_count, self._population_size)    

    
    def __display_population_fitness(self, population):
        best_individual = super()._get_best_individual(population)
        average_fitness = round(self.__get_average_fitness(population), 2)
        print("[Gen] {} [Average Fitness of Population] {} [Best Individual] {}".format(
            self._generation, average_fitness, best_individual.fitness))
        self._generation_info.append([self._generation, average_fitness, best_individual.fitness])
    

    def __reset_population_fitness(self, population):
        for individual in population:
            individual.fitness = 0        


    def __get_average_fitness(self, population):
        fitness_total = 0
        for individual in population:
            fitness_total += individual.fitness
        
        return fitness_total / self._population_size
     

    def __generate_offspring(self, population):
        best_individual = self._get_best_individual(population)
        offspring = tournament_selection(deepcopy(population))
        offspring = two_point_crossover(offspring, self._crossover_probability,
            self._chromosome_size)

        for individual in offspring:
            rule_based_mutate(individual, 
                self._mutation_probabilty, self._rule_size)

        worst_ind, worst_idx = self._get_worst_individual(population=self._population,
            with_index=True)

        offspring[worst_idx] = best_individual
        return offspring
    

    def __test_rules(self, rule_set):
        total_rules_matched = 0
        for data in self._data:
            for rule in rule_set:
                    # Rule has matched
                    if(data.label == str(rule.prediction)):
                        # The rule is good
                        print("Rule {} {} matched {} {} successfully".format(rule.feature, rule.prediction, data.feature, data.label))
                        total_rules_matched += 1
                        break
                    else:
                        print("Rule {} matched data {} but gave the wrong result.".format(rule.feature, data.feature))
                        break
        print("[TOTAL DATA POINTS CORRECTLY MATCHED => {}]".format(total_rules_matched))
        


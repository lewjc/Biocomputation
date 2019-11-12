from classes.rule_based_ga import RuleBasedGeneticAlgorithm
from modules.selection import tournament_selection
from modules.crossover import do_one_point_crossover, do_two_point_crossover
from modules.mutation import rule_based_mutate
from modules.data import draw_graph
from classes.rule import Rule
from copy import deepcopy
import random

class AdaptiveRuleBasedGeneticAlgorithm(RuleBasedGeneticAlgorithm):
    
    def __init__(self, dataset: list, population_size=10,
        mutation_probability=0.005, rule_count=10):

        super().__init__(dataset, population_size=population_size, 
            mutation_probability = mutation_probability, rule_count=rule_count)
            
        self.CROSSOVER_CONST_ONE = 1
        self.CROSSOVER_CONST_TWO = 1
        self.MUTATION_CONST_ONE = 0.5
        self.MUTATION_CONST_TWO = 0.5


    def evolve(self, epoch=1000):

        while(self._generation < epoch):
            self.__evaluate_population_fitness(self._population)
            self._display_population_fitness(self._population)
            self._population = self._generate_offspring(self._population)
            best = self._get_best_individual(self._population)
            if(best.fitness == self._max_fitness):
                break
            self._reset_population_fitness(self._population)
            self._generation +=1 
        
        rules = Rule.generate_rules_from_chromosome(best.chromosome, self._feature_size, 
            self._rule_size)

        print("[BEST RULE BASE]")
        for rule in rules:
            print("RULE: {} {}".format(rule.feature, rule.label))

        draw_graph(self._generation_info, epoch, "Adaptive",
            "Adaptive", self._rule_count, self._population_size)    


    def __evaluate_population_fitness(self, population):
          for individual in population:
                self._evaluate_fitness(individual)
        

    def _generate_offspring(self, population):
        best_individual = self._get_best_individual(population)
        average_fitness_of_population = self._get_average_fitness(population)
        offspring = tournament_selection(deepcopy(population))
        offspring = self.__adaptive_crossover(offspring, average_fitness_of_population)
        self.__adaptive_mutation(offspring, average_fitness_of_population)        
        _, worst_idx = self._get_worst_individual(population=offspring,
            with_index=True)
        offspring[worst_idx] = best_individual

        return offspring
    
    def __adaptive_crossover(self, population, average_fitness):
        new_population = []
        for i in range(0, self._population_size, 2):
            parent_one = population[i]
            parent_two = population[i+1]
            best_parent = (parent_one if 
                parent_one.fitness > parent_two.fitness else parent_two)
            crossover_probability = self.__calculate_crossover_probability(best_parent.fitness,
            self._max_fitness, average_fitness)
            chance = random.uniform(0, 1)
            if(chance <= crossover_probability):
                parent_one, parent_two = do_one_point_crossover(parent_one, parent_two)

            new_population.append(parent_one)
            new_population.append(parent_two)

        return new_population

    def __adaptive_mutation(self, population, average_fitness):
        for individual in population:
            mutation_probability = self.__calculate_mutation_probability(
                individual.fitness, self._max_fitness, average_fitness)
            rule_based_mutate(individual, 
                mutation_probability, self._rule_size)


    def __calculate_crossover_probability(self, best_parent_fitness, max_fitness, 
        average_population_fitness):

        if(best_parent_fitness >= average_population_fitness):
            calc = ((max_fitness - best_parent_fitness) / (
                max_fitness - average_population_fitness))
            calc = calc * 1/max_fitness
            return (self.CROSSOVER_CONST_ONE * calc)
        else:
            return self.CROSSOVER_CONST_TWO
            
    
    def __calculate_mutation_probability(self, individual_fitness, max_fitness, 
        average_population_fitness):
        
        if(individual_fitness >= average_population_fitness):
            calc = ((max_fitness - individual_fitness) / 
                (max_fitness - average_population_fitness))
            calc = calc * 1/max_fitness
            return(self.MUTATION_CONST_ONE * calc)
        else:
            return self.x
    
    def __adaptive_population(self, best_fitness, 
        average_fitness, current_population_size):
        new_pop_size = (current_population_size + (best_fitness / average_fitness) - 1)
        print("[NEW POPULATION SIZE] = {}".format(new_pop_size))
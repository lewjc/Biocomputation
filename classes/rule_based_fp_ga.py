from classes.base_ga import GeneticAlgorithmBase
from modules.data import initialise_floating_point_data
from classes.individual import Individual
from classes.rule import Rule
from modules.selection import tournament_selection
from modules.crossover import one_point_crossover, two_point_crossover
from modules.mutation import floating_point_boundary_mutate
from modules.data import draw_graph
from copy import deepcopy
import random
import uuid


class RuleBasedFloatingPointGA(GeneticAlgorithmBase):

    def __init__(self, dataset, test_data, population_size=50,
                 mutation_probability=0.001, crossover_probability=0.8, rule_count=10,
                 seed_edge_cases=False):

        super().__init__(population_size=population_size,
                         mutation_probability=mutation_probability,
                         crossover_probability=crossover_probability)

        self._rule_count = rule_count
        (self._train_data, self._feature_size,
            self._label_size) = initialise_floating_point_data(dataset)
        self._test_data = initialise_floating_point_data(test_data)[0]
        self._rule_size = self._feature_size + self._label_size
        self._chromosome_size = self._rule_size * self._rule_count
        self._seed_edge_cases = seed_edge_cases
        self._initialise_population()
        self._generation_info = []
        self._max_fitness = len(self._train_data)

    def _initialise_population(self):
        for i in range(0, self._population_size):
            chromosome = self._generate_chromosome()
            self._population.insert(i,
                                    Individual(uuid.uuid1().hex, chromosome, 0))

    def _generate_chromosome(self):
        chromosome = []
        while len(chromosome) < self._chromosome_size:
            for i in range(self._feature_size):
                def gene(): return random.random()
                bounds = list((0, 1)) if self._seed_edge_cases else list(
                    (gene(), gene()))
                chromosome.append((min(bounds), max(bounds)))

            for j in range(self._label_size):
                chromosome.append(random.choice([0, 1]))

        return chromosome

    def evolve(self, epoch=1500):
        while(self._generation < epoch):
            for individual in self._population:
                self._evaluate_fitness(individual)
            self._display_population_fitness(self._population)
            self._population = self._generate_offspring(self._population)
            best = self._get_best_individual(self._population)
            if(best.fitness == self._max_fitness):
                break
            self._reset_population_fitness(self._population)

            if(self._generation != 0 and (self._generation + 1) % 5 == 0):
                self.evalute_test(self._population)
                self._reset_population_fitness(self._population)

            self._generation += 1

        rules = Rule.generate_rules_from_chromosome(best.chromosome, self._feature_size,
                                                    self._rule_size)

        self.evalute_test(self._population)
        print("[BEST RULE BASE]\n")
        for rule in rules:
            print("RULE: {} {}".format(rule.feature, rule.label))

        draw_graph(self._generation_info, epoch, self._crossover_probability,
                   self._mutation_probabilty, self._rule_count, self._population_size, len(self._test_data))
        self.test_rules(rules)
        self.floating_point_to_rules(rules)

    def evalute_test(self, population):
        for individual in population:
            self._evaluate_fitness(individual, use_test=True)
        best_individual = self._get_best_individual(population)
        average_fitness = round(self._get_average_fitness(population), 2)
        print("[Gen] {} [Test Average] {} [Test Best] {}".format(
            self._generation, average_fitness, best_individual.fitness))
        for i in range(self._generation - 5, self._generation):
            self._generation_info[i][-1], self._generation_info[i][-2] = (
                average_fitness, best_individual.fitness)

    def _evaluate_fitness(self, individual, use_test=False):
        rules = Rule.generate_rules_from_chromosome(individual.chromosome,
                                                    self._feature_size, self._rule_size)

        dataset = self._test_data if use_test else self._train_data
        for data in dataset:
            for rule in rules:
                if(self._does_rule_match(data.features, rule.feature)):
                    if(data.prediction == rule.label):
                        individual.fitness += 1
                    break

    def _does_rule_match(self, data, rule):
        return all(b <= a <= c for a, (b, c) in zip(data, rule))

    def _display_population_fitness(self, population):
        best_individual = self._get_best_individual(population)
        average_fitness = round(self._get_average_fitness(population), 2)
        print("[Gen] {} [Average] {} [Best] {}".format(
            self._generation, average_fitness, best_individual.fitness))
        self._generation_info.append(
            [self._generation, average_fitness, best_individual.fitness, 0, 0])

    def _generate_offspring(self, population):
        best_individual = self._get_best_individual(population)
        offspring = tournament_selection(population)
        offspring = two_point_crossover(offspring, self._crossover_probability,
                                        self._chromosome_size)

        for individual in offspring:
            floating_point_boundary_mutate(individual,
                                           self._mutation_probabilty, self._rule_size)
            # individual.chromosome = (individual.chromosome[self._rule_size:] +
            #                          individual.chromosome[:self._rule_size])

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
            return self.MUTATION_CONST_TWO

    def test_rules(self, finished_ruleset):
        overall_fitness = 0
        failed = 0
        for data in self._test_data:
            data_matched = False
            for rule in finished_ruleset:
                if(self._does_rule_match(data.features, rule.feature) and
                        data.prediction == rule.label):
                    overall_fitness += 1
                    data_matched = True
                    break
            if(not data_matched):
                failed += 1

        percentage_passed = (overall_fitness / len(self._train_data)) * 100
        print("TEST COMPLETE: PASSED: {} FAILED: {} | OVERALL: {}".format(
            overall_fitness, failed, percentage_passed))

    def floating_point_to_rules(self, finished_ruleset):

        for rule in finished_ruleset:

            print("RULE: {} ----- {}".format(
                [(lb, ub) if ((ub - lb) < 0.8) else ('#', '#')
                 for (lb, ub) in rule.feature], rule.label))

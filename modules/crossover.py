import random
from copy import deepcopy

def one_point_crossover(population: list, 
    crossover_probability: float, chromosome_size: int):

    population = deepcopy(population)
    crossed_over_population = []     
    crossover = lambda i1, i2, point: i1.chromosome[:point].copy() + i2.chromosome[point:].copy()
    population_size = len(population)
    for i in range(0, population_size, 2):
        chance = random.uniform(0, 1)
        do_crossover = (chance <= crossover_probability)
        parent_one = population[i]            
        parent_two = population[(i + 1)]
        
        if(do_crossover):
            crossover_point = random.randrange(chromosome_size)
        
            parent_one_chromosome = crossover(parent_one,
                parent_two, crossover_point)

            parent_two_chromosome = crossover(parent_two,
                parent_one, crossover_point)

            parent_one.chromosome = parent_one_chromosome
            parent_two.chromosome = parent_two_chromosome
        
        crossed_over_population.append(parent_one)
        crossed_over_population.append(parent_two)
        
    return crossed_over_population

def two_point_crossover(population: list, crossover_probability: float,
    chromosome_size, invert=False):
    population = deepcopy(population)
    crossed_over_population = []     
    crossover = lambda i1, i2, p1, p2: (i1.chromosome[:p1].copy() + 
        i2.chromosome[p1:p2].copy() + i1.chromosome[p2:].copy())
    population_size = len(population)
    for i in range(0, population_size, 2):
        chance = random.uniform(0, 1)
        do_crossover = (chance <= crossover_probability)
        parent_one = population[i]            
        parent_two = population[(i + 1)]
        
        if(do_crossover):
            crossover_point_one = 0
            crossover_point_two = 0

            while((crossover_point_one) == (crossover_point_two)):
                crossover_point_one = random.randrange(chromosome_size)
                crossover_point_two = random.randrange(chromosome_size)

            biggest = (crossover_point_one if crossover_point_one > crossover_point_two
                else crossover_point_two)

            smallest = (crossover_point_one if crossover_point_two == biggest 
                else  crossover_point_two)

            parent_one_chromosome = crossover(parent_one,
                parent_two, smallest, biggest)

            parent_two_chromosome = crossover(parent_two,
                parent_one, smallest, biggest)

            parent_one.chromosome = parent_one_chromosome
            parent_two.chromosome = parent_two_chromosome
        
        crossed_over_population.append(parent_one)
        crossed_over_population.append(parent_two)
        
    return crossed_over_population

def uniform_crossover(population, crossover_probability):
    crossed_over_population = []
    for i in range(0, population, 2):
        parent_one = population[i]
        parent_two = population[i+1]
        chance = random.uniform(0, 1)
        do_crossover = (chance <= crossover_probability)

        if(do_crossover):                    
            parent_one_chromosome = uniform_inner(parent_one, parent_two)
            parent_two_chromosome = uniform_inner(parent_one, parent_two)
            parent_one.chromosome = parent_one_chromosome
            parent_two.chromosome = parent_two_chromosome

        crossed_over_population.append(parent_one)
        crossed_over_population.append(parent_two)

    return crossed_over_population


def uniform_inner(parent_one, parent_two):
    new = []
    for i in range(0, len(parent_one.chromosome)):
            chance = random.randint(0, 1)
            if(chance == 0):
                new.insert(i, parent_one.chromosome[i])
            else:
                new.insert(i, parent_two.chromosome[i])
    return new

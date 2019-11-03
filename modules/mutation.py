import random
from Assignment.Individual import Individual

def bit_flip_mutate(individual: Individual, mutation_probabilty: float):
        idx = -1
        for cell in individual.chromosome:
            idx += 1
            chance = random.uniform(0, 1)
            mutate = (chance <= mutation_probabilty)
            if(mutate):       
                cell ^= 1
                individual.chromosome[idx] = cell                
            else:
                continue

def rule_based_mutate(individual: Individual, 
    mutation_probability: float, rule_size: int):
    idx = -1
    for cell in individual.chromosome:
        idx += 1
        chance = random.uniform(0, 1)
        mutate = (chance <= mutation_probability)
        if(mutate):                   
            possibilities = [0,1,'#']
            if ((idx + 1) % rule_size == 0):
                possibilities = [0,1]                
            possibilities.remove(cell)
            cell = random.choice(possibilities)
            
            individual.chromosome[idx] = cell            